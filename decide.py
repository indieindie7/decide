"""decide.py: "System 1" option picks without JSON (idea: Codacus / Harsha "Jev mode", video bcGO7xre46o; 10-15).

Every field is asked as "<field>:" after one shared prompt and the model gives ONE token: the letter of an option.
Fields must not depend on each other.  What the 10-15 tests found (test_<backend>.json):
- llama-server (Gemma 4 12B): the confidence is the letter's share of the allowed letters' probability.  One slot,
  cached prefix: 590 ms/state vs 1191 ms for writing grammar JSON; 8 parallel slots gave nothing (each re-reads it).
- The prompt must end "field:" with NO trailing space (the model's token is " B"), and thinking must be off.
- The shared part goes first (instructions + fields + options), the state last, so the cache carries across states.
- Strata has no logprobs: logit_bias +100 on the letter tokens + max_tokens 1 gives the pick but no confidence;
  "strata_prefix" pins the shared messages.  10-15 rerun: llama 429 vs 974 ms/state, agree 57/80; Strata 3035 vs
  1534 ms (each field request re-prefills), agree 42/80 and leans to "A" -> on Strata, keep writing JSON.

10-15 upgrades (bycloud "Decision Models", video-notes tree/36), llama backend only:
- --mode text ("Open Jev"): each whole option string is scored as a continuation (mean token logprob, 1-token
  steps on the cached prefix) instead of one letter; for options that are commands, names, tool calls.
- calib: rule-labelled states (CAL_SYSTEM), one temperature fitted on half (calib.py), Brier/ECE on the other half,
  auto threshold = lowest conf with >= --target accuracy -> calib_<name>_<mode>.json; decide() then returns
  (option, calibrated conf, route auto|ask|escalate).

  py -3.13 decide.py test [--backend llama|strata] [--model gemma|mimo] [--n 20]
  py -3.13 decide.py calib [--mode letter|text] [--n 120] [--target 0.95]   # llama only
  py -3.13 decide.py run task.json [--backend ...]      # {"system", "schema": {field: [options]}, "states": [...]}
  from decide import Llama, Strata; b = Llama(); b.decide(system, schema, state) -> {field: (option, conf|None)}
Take the GPU token first (gpu_queue.py req "<chat>: decide").
"""
import argparse, json, math, os, random, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
try:
    import localread as lr                       # local helper (server path + health check)
except ImportError:                              # standalone: LLAMA_SERVER=<llama-server.exe>, MODELS_DIR=<gguf folder>
    import types, urllib.request
    def _healthy(port):
        try:
            return urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).status == 200
        except Exception:
            return False
    lr = types.SimpleNamespace(SERVER=os.environ.get("LLAMA_SERVER", "llama-server"), healthy=_healthy,
                               HERE=os.path.dirname(os.environ.get("MODELS_DIR", os.path.join(HERE, "models"))))
import calib

LETTERS = "ABCDEFGH"
RULE_TEXT = ("Answer with one line per field: the field name, a colon and the option text exactly as listed, "
             "e.g. \"action: block\". Nothing else.")
RULE = ("Answer with one line per field: the field name, a colon and ONLY the option letter, "
        "e.g. \"action: B\". No words, no numbers.")
NOTHINK = {"enable_thinking": False}


def shared_text(system, schema, mode="letter"):
    if mode == "text":
        lines = [f"{f}: " + " | ".join(opts) for f, opts in schema.items()]
        return system + "\n\nFields and options:\n" + "\n".join(lines) + "\n\n" + RULE_TEXT
    lines = [f"{f}: " + ", ".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(opts)) for f, opts in schema.items()]
    return system + "\n\nFields and options:\n" + "\n".join(lines) + "\n\n" + RULE


class Llama:
    """Stock llama-server, one slot, logprobs -> pick + confidence."""
    PORT = 8086
    MODELS = {"gemma": "gemma-4-12b-it-qat-q4_0.gguf", "mimo": "MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf"}

    def _start(self, model):
        self.name = f"llama-{model}"
        log = open(os.path.join(HERE, "server.log"), "w")
        self.p = subprocess.Popen([lr.SERVER, "-m", os.path.join(os.environ.get("MODELS_DIR", os.path.join(lr.HERE, "models")), self.MODELS[model]),
                                   "--port", str(self.PORT), "-ngl", "99", "-c", "8192", "-np", "1", "-fa", "on",
                                   "--jinja"], cwd=os.path.dirname(lr.SERVER) or None, stdout=log, stderr=subprocess.STDOUT,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for _ in range(180):
            if lr.healthy(self.PORT):
                return
            if self.p.poll() is not None:
                sys.exit("llama-server exited, see server.log")
            time.sleep(1)
        self.close(); sys.exit("llama-server did not come up")

    def close(self):
        self.p.kill()

    def __init__(self, model="gemma", mode="letter"):
        self.mode, self.cal = mode, calib.load(os.path.join(HERE, f"calib_llama-{model}_{mode}.json"))
        self._start(model)

    def prefix(self, system, schema, state):
        return lr.http(self.PORT, "/apply-template", {"messages": [
            {"role": "system", "content": shared_text(system, schema, self.mode)},
            {"role": "user", "content": f"State:\n{state}"}], "chat_template_kwargs": NOTHINK})["prompt"]

    def probs(self, system, schema, state):
        """{field: {option: p}}, raw (uncalibrated) model probabilities."""
        prefix = self.prefix(system, schema, state)
        return {f: (self._letter_dist if self.mode == "letter" else self._text_dist)(prefix, f, opts)
                for f, opts in schema.items()}

    def _letter_dist(self, prefix, f, opts):
        r = lr.http(self.PORT, "/completion", {"prompt": prefix + f"{f}:", "n_predict": 1, "n_probs": 100,
                                               "temperature": 0, "cache_prompt": True})
        probs = dict.fromkeys(opts, 0.0)
        for c in r.get("completion_probabilities", [{}])[0].get("top_logprobs", []) or []:
            tok = c.get("token", "").strip().rstrip(")").upper()
            if len(tok) == 1 and tok in LETTERS[:len(opts)]:
                probs[opts[LETTERS.index(tok)]] += math.exp(c["logprob"])
        z = sum(probs.values())
        return {o: p / z for o, p in probs.items()} if z else probs

    def _tok(self, text):
        return lr.http(self.PORT, "/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]

    def _text_dist(self, prefix, f, opts):
        """"Open Jev" mode: score each whole option as a continuation of the cached prefix (teacher forcing)."""
        head = self._tok(prefix + f"{f}:")
        lps = {}
        for o in opts:
            full = self._tok(prefix + f"{f}: {o}\n")
            k0 = next((i for i, (x, y) in enumerate(zip(head, full)) if x != y), len(head))
            seq = []
            for k in range(k0, len(full)):            # one 1-token step per option token; the prefix stays cached
                r = lr.http(self.PORT, "/completion", {"prompt": full[:k], "n_predict": 1, "n_probs": 100,
                                                       "temperature": 0, "cache_prompt": True})
                top = r.get("completion_probabilities", [{}])[0].get("top_logprobs", []) or []
                hit = [c["logprob"] for c in top if c.get("id") == full[k]]
                seq.append(hit[0] if hit else min((c["logprob"] for c in top), default=-30.0) - 1.0)
            lps[o] = seq
        return calib.seq_dist(lps, norm="sum")             # sum = real sequence probability; mean favoured long options (10-15: 54% acc)

    def decide(self, system, schema, state):
        """{field: (option, calibrated conf, route)}; route = auto | ask | escalate (calib_<name>_<mode>.json)."""
        out = {}
        for f, d in self.probs(system, schema, state).items():
            if not sum(d.values()):                     # no option seen: unreadable, not "first option"
                out[f] = (None, 0.0, "escalate"); continue
            o, p = calib.top(calib.apply_temp(d, self.cal.get("T_field", {}).get(f, self.cal["T"])))
            out[f] = (o, p, calib.route(p if o else None, self.cal["auto"], self.cal["ask"]))
        return out

    def write_json(self, system, schema, state):
        return chat_json(self.PORT, system, schema, state)


class Strata:
    """Strata on 8090: no logprobs, so logit_bias forces a letter; the pick only (conf None)."""
    LETTER_IDS = {c: (32 + i, [357, 417, 351, 414, 458, 426, None, None][i]) for i, c in enumerate("ABCDEF")}
    # ids from Strata's iq3_xxs tokenizer vocab.json: "A".."F" = 32..37, "ĠA".."ĠF" above

    def __init__(self, model="strata"):
        self.name = model
        self.own, self.PORT = lr.open_server(model)

    def close(self):
        if self.own is not None:                    # stop by port: the server is two python.exe processes
            subprocess.run(["powershell", "-NoProfile", "-Command",
                            f"Get-NetTCPConnection -LocalPort {self.PORT} -State Listen -EA 0 | % {{ Stop-Process "
                            "-Id $_.OwningProcess -Force -EA 0 }"], creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def decide(self, system, schema, state):
        base = [{"role": "system", "content": shared_text(system, schema)},
                {"role": "user", "content": f"State:\n{state}"}]
        out = {}
        for f, opts in schema.items():
            bias = {str(i): 100 for c in LETTERS[:len(opts)] for i in self.LETTER_IDS[c] if i is not None}
            r = lr.http(self.PORT, "/v1/chat/completions", {
                "messages": base + [{"role": "user", "content": f"{f}:"}], "max_tokens": 1, "temperature": 0,
                "logit_bias": bias, "chat_template_kwargs": NOTHINK, "strata_prefix": {"messages": 1}})
            tok = (r["choices"][0]["message"].get("content") or "").strip().upper()[:1]
            out[f] = (opts[LETTERS.index(tok)] if tok and tok in LETTERS[:len(opts)] else None, None)
        return out

    def write_json(self, system, schema, state):
        return chat_json(self.PORT, system, schema, state)


def chat_json(port, system, schema, state):
    """Baseline: the model writes the whole JSON object (llama-server: grammar; Strata: validated once)."""
    props = {f: {"type": "string", "enum": opts} for f, opts in schema.items()}
    user = f"State:\n{state}\n\nReply with JSON only, one value per field, from these options:\n" + \
        "\n".join(f"{f}: {', '.join(o)}" for f, o in schema.items())
    try:
        r = lr.http(port, "/v1/chat/completions", {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0, "max_tokens": 400, "chat_template_kwargs": NOTHINK,
            "response_format": {"type": "json_schema", "json_schema": {"name": "d", "schema": {
                "type": "object", "properties": props, "required": list(schema)}}}})
        txt = r["choices"][0]["message"].get("content") or ""
        return json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
    except Exception as e:
        open(os.path.join(HERE, "json_fail.txt"), "a", encoding="utf-8").write(repr(e)[:300] + "\n")
        return {}


def backend(a):
    return Strata(a.model if a.model in ("strata", "coder") else "strata") if a.backend == "strata" else Llama(a.model, a.mode)


SYSTEM = "You control one enemy in a side-scrolling beat-'em-up. Pick sensible, fair moves for the situation."
SCHEMA = {"action": ["attack", "block", "retreat", "circle", "taunt", "grab"],
          "target": ["player", "nearest ally", "pickup item", "none"],
          "stance": ["aggressive", "cautious", "panicked"],
          "call_for_help": ["yes", "no"]}


def states(n, seed=7):
    rnd = random.Random(seed)
    return [f"enemy hp {rnd.randint(5, 100)}%, player hp {rnd.randint(5, 100)}%, distance "
            f"{rnd.choice(['touching', 'close', 'mid', 'far'])}, player is "
            f"{rnd.choice(['attacking', 'blocking', 'stunned', 'walking away', 'holding a pipe'])}, "
            f"allies nearby {rnd.randint(0, 3)}, health pickup {rnd.choice(['on the floor nearby', 'none'])}, "
            f"enemy type {rnd.choice(['grunt', 'brute', 'knife thug'])}" for _ in range(n)]


def test(a):
    b = backend(a)
    try:
        S = states(a.n)
        b.decide(SYSTEM, SCHEMA, S[0])                                  # warm-up
        t = time.time(); D = [b.decide(SYSTEM, SCHEMA, s) for s in S]; td = time.time() - t
        t = time.time(); J = [b.write_json(SYSTEM, SCHEMA, s) for s in S]; tj = time.time() - t
    finally:
        b.close()
    F = [(i, f) for i in range(len(S)) for f in SCHEMA]
    same = sum(D[i][f][0] == J[i].get(f) for i, f in F)
    conf = lambda ok: [D[i][f][1] for i, f in F if (D[i][f][0] == J[i].get(f)) == ok and D[i][f][1] is not None]
    avg = lambda x: round(sum(x) / len(x), 3) if x else None
    rep = {"backend": b.name, "states": len(S), "fields": len(F),
           "decide_ms_per_state": round(td / len(S) * 1000), "json_ms_per_state": round(tj / len(S) * 1000),
           "agreement": f"{same}/{len(F)}", "json_empty": sum(not j for j in J),
           "unreadable": sum(D[i][f][0] is None for i, f in F),
           "mean_conf_agree": avg(conf(True)), "mean_conf_disagree": avg(conf(False)),
           "examples": [{"state": S[i], "decide": D[i], "json": J[i]} for i in range(3)]}
    json.dump(rep, open(os.path.join(HERE, f"test_{b.name}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "examples"}))


def run(a):
    task = json.load(open(a.task, encoding="utf-8"))
    b = backend(a)
    try:
        for s in task["states"]:
            print(json.dumps({"state": s, "decide": b.decide(task["system"], task["schema"], s)}), flush=True)
    finally:
        b.close()


CAL_SYSTEM = SYSTEM + """ Follow these rules exactly (hp = enemy hp):
- stance: hp 60 or more = aggressive; 30 to 59 = cautious; under 30 = panicked.
- call_for_help: yes only if allies nearby is 1 or more AND hp is under 40; else no.
- target: pickup item if a health pickup is on the floor nearby AND hp is under 50; else player.
- action: hp under 30 = retreat; else player stunned = grab; else player attacking = block;
  else distance touching or close = attack; else circle."""


def labelled_states(n, seed=11):
    """States with rule answers (ground truth for calibration; the old 80 only measured JSON agreement)."""
    rnd, out = random.Random(seed), []
    for _ in range(n):
        hp, php, al = rnd.randint(5, 100), rnd.randint(5, 100), rnd.randint(0, 3)
        dist = rnd.choice(["touching", "close", "mid", "far"])
        pl = rnd.choice(["attacking", "blocking", "stunned", "walking away", "holding a pipe"])
        pick = rnd.choice(["on the floor nearby", "none"])
        text = (f"enemy hp {hp}%, player hp {php}%, distance {dist}, player is {pl}, allies nearby {al}, "
                f"health pickup {pick}, enemy type {rnd.choice(['grunt', 'brute', 'knife thug'])}")
        act = ("retreat" if hp < 30 else "grab" if pl == "stunned" else "block" if pl == "attacking"
               else "attack" if dist in ("touching", "close") else "circle")
        out.append((text, {"action": act, "target": "pickup item" if pick != "none" and hp < 50 else "player",
                           "stance": "aggressive" if hp >= 60 else "cautious" if hp >= 30 else "panicked",
                           "call_for_help": "yes" if al and hp < 40 else "no"}))
    return out


def calibrate(a):
    if a.backend != "llama":
        sys.exit("calib needs the llama backend (Strata has no logprobs)")
    """Fit one temperature on half the labelled states, report the other half, save calib_<name>_<mode>.json."""
    b = backend(a)
    try:
        L = labelled_states(a.n)
        b.probs(CAL_SYSTEM, SCHEMA, L[0][0])                              # warm-up
        t = time.time(); P = [b.probs(CAL_SYSTEM, SCHEMA, s) for s, _ in L]; ms = (time.time() - t) / len(L) * 1000
    finally:
        b.close()
    S = [[(P[i][f], L[i][1][f]) for f in SCHEMA] for i in range(len(L))]
    unread = sum(not sum(d.values()) for row in S for d, _ in row)
    fit = [x for row in S[: len(S) // 2] for x in row if sum(x[0].values())]; ev = [x for row in S[len(S) // 2:] for x in row]
    T = calib.fit_temp(fit)
    # one temperature per field (refit.py on run 2: pooled ece 0.083 -> 0.067, global T made it worse)
    Tf = {f: calib.fit_temp([(P[i][f], L[i][1][f]) for i in range(len(L) // 2) if sum(P[i][f].values())]) for f in SCHEMA}
    fitc = [(calib.apply_temp(P[i][f], Tf[f]), L[i][1][f]) for i in range(len(L) // 2) for f in SCHEMA if sum(P[i][f].values())]
    auto = calib.auto_threshold(fitc, 1.0, target_acc=a.target)
    cal = {"T": T, "T_field": Tf, "auto": auto if auto is not None else 1.01, "ask": 0.6}
    evf = [f for row in S[len(S) // 2:] for f in SCHEMA]                 # field of each ev sample
    routes = [calib.route(calib.top(calib.apply_temp(d, Tf[f]))[1], cal["auto"], cal["ask"]) for (d, _), f in zip(ev, evf)]
    acc_of = lambda r: [calib.top(d)[0] == y for (d, y), k in zip(ev, routes) if k == r]
    rep = {"backend": b.name, "mode": a.mode, "states": len(L), "unreadable": unread, "ms_per_state": round(ms), "fit_n": len(fit),
           "eval_n": len(ev), "T": round(T, 3), "auto_threshold": cal["auto"], "target_acc": a.target,
           "T_field": {f: round(v, 3) for f, v in Tf.items()},
           "eval_raw": calib.report(ev, 1.0), "eval_calibrated": calib.report(ev, T),
           "eval_field_T": calib.report([(calib.apply_temp(d, Tf[f]), y) for (d, y), f in zip(ev, evf)], 1.0),
           "routes": {r: {"share": round(len(acc_of(r)) / len(ev), 3),
                          "acc": round(sum(acc_of(r)) / len(acc_of(r)), 3) if acc_of(r) else None}
                      for r in ("auto", "ask", "escalate")},
           "per_field_acc": {f: round(sum(calib.top(P[i][f])[0] == L[i][1][f] for i in range(len(L))) / len(L), 3)
                             for f in SCHEMA}}
    calib.save(os.path.join(HERE, f"calib_{b.name}_{a.mode}.json"), cal)
    json.dump({"probs": P, "labels": [l for _, l in L]}, open(os.path.join(HERE, f"calibraw_{b.name}_{a.mode}.json"), "w"))
    json.dump(rep, open(os.path.join(HERE, f"calibrep_{b.name}_{a.mode}.json"), "w"), indent=1)
    print(json.dumps(rep))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["test", "run", "calib"]); ap.add_argument("task", nargs="?")
    ap.add_argument("--backend", default="llama", choices=["llama", "strata"])
    ap.add_argument("--model", default="gemma"); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--mode", default="letter", choices=["letter", "text"])
    ap.add_argument("--target", type=float, default=0.95)
    a = ap.parse_args()
    {"test": test, "run": run, "calib": calibrate}[a.cmd](a)

// klein - Qwen3.8 inference engine
// Copyright (C) 2026 klein contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// klein command line:
//   klein run   -m model.gguf -p "prompt"          chat completion printed to stdout
//   klein bench -m model.gguf [-pp 512] [-n 128]   prefill and decode speed
//   klein ppl   -m model.gguf -f text.txt [-c 512] perplexity (same scoring as llama.cpp's llama-perplexity)
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"
#include "engine.h"
#include "server.h"

using namespace klein;

namespace {

struct Args {
    std::string cmd;
    EngineConfig ec;
    std::string prompt, file;
    int n_predict = 256;
    int pp = 512;
    int ppl_ctx = 512;
    int ppl_chunks = 0;
    int ppl_batch = 64;
    std::string kld_base;
    ServerOptions so;
    bool raw = false;
    bool think = false;
    SamplerParams sp;
};

void usage() {
    std::fprintf(stderr,
        "usage: klein <command> -m MODEL.gguf [options]\n"
        "\n"
        "commands:\n"
        "  run          generate a reply to -p TEXT / -f FILE (Qwen chat format unless --raw)\n"
        "  serve        OpenAI-compatible HTTP server (/v1/chat/completions, /v1/completions, /completion)\n"
        "  bench        prefill + decode speed on a synthetic prompt (-pp N tokens, -n N generated)\n"
        "  ppl          perplexity of -f FILE (llama-perplexity scoring); --kld-base FILE for KL divergence\n"
        "  check-spec   verify that speculative decoding emits what plain decoding emits (greedy)\n"
        "\n"
        "model and memory:\n"
        "  --mtp FILE          GGUF with a (smaller) MTP block to use for drafting instead of the model's own\n"
        "  -c, --ctx N         context length (default: the model's, 262144)\n"
        "  --kv TYPE           KV cache type: auto|f16|q8_0|q4_0 (auto: q4_0 above 128K, else q8_0)\n"
        "  --kv-place P        auto|gpu|host (auto: RAM unless the cache is small or everything fits)\n"
        "  --kv-cpu-attn       with the KV cache in RAM, attend on the CPU instead of zero-copy GPU reads\n"
        "  --vram-margin MB    VRAM left unused for the driver and CUDA pool (default 256)\n"
        "  --compute-reserve MB  VRAM reserved for decode compute buffers (default 192)\n"
        "  --no-repack         no repacked CPU copies of spilled weights (less RAM, slower decode)\n"
        "  -t N                CPU threads (default: 3/4 of the hardware threads)\n"
        "  -ub N               prefill chunk in tokens (default 2048)\n"
        "\n"
        "speculative decoding (MTP):\n"
        "  --draft N           draft tokens per step (default 3; 0 = off). 3 gives verify batches of 4, which\n"
        "                      the repacked AVX2 kernels process in one pass\n"
        "  --mtp-window N      MTP attention window in positions (default 16384)\n"
        "  --snap-type T       DeltaNet rollback snapshots: f16 (default), bf16 or f32\n"
        "\n"
        "run / check-spec:\n"
        "  -p TEXT | -f FILE   prompt\n"
        "  -n N                tokens to generate (default 256)\n"
        "  --raw               no chat template; --think: leave thinking on (default: off for run)\n"
        "  --temp T --top-k K --top-p P --min-p P --presence-penalty P --seed S   sampling (default greedy)\n"
        "\n"
        "serve:\n"
        "  --host H (default 127.0.0.1) --port N (default 8080) --api-key KEY --alias NAME\n"
        "  a non-loopback --host requires --api-key\n"
        "\n"
        "bench:   -pp N (prompt tokens, default 512) -n N (generated tokens)\n"
        "ppl:     -f FILE --ppl-ctx N (default 512) --chunks N --ppl-batch N (default 64) --kld-base FILE\n"
        "other:   -v (debug log)\n");
    std::exit(1);
}

Args parse(int argc, char** argv) {
    if (argc < 2) usage();
    Args a;
    a.cmd = argv[1];
    a.sp.temperature = 0.0f;
    for (int i = 2; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) usage();
            return argv[++i];
        };
        if (s == "-m") a.ec.model_path = next();
        else if (s == "--mtp") a.ec.mtp_path = next();
        else if (s == "-c" || s == "--ctx") a.ec.n_ctx = std::stoi(next());
        else if (s == "--kv") a.ec.kv_type = next();
        else if (s == "--kv-place") a.ec.kv_place = next();
        else if (s == "-t") a.ec.n_threads = std::stoi(next());
        else if (s == "-ub") a.ec.n_ubatch = std::stoi(next());
        else if (s == "--draft") a.ec.n_draft = std::stoi(next());
        else if (s == "--vram-margin") a.ec.vram_margin_mb = std::stoul(next());
        else if (s == "--compute-reserve") a.ec.compute_reserve_mb = std::stoul(next());
        else if (s == "--no-repack") a.ec.cpu_repack = false;
        else if (s == "--kv-cpu-attn") a.ec.kv_zero_copy = false;
        else if (s == "--mtp-window") a.ec.mtp_window = std::stoi(next());
        else if (s == "--snap-type") a.ec.snap_type = next();
        else if (s == "-p") a.prompt = next();
        else if (s == "-f") a.file = next();
        else if (s == "-n") a.n_predict = std::stoi(next());
        else if (s == "-pp") a.pp = std::stoi(next());
        else if (s == "--ppl-ctx") a.ppl_ctx = std::stoi(next());
        else if (s == "--chunks") a.ppl_chunks = std::stoi(next());
        else if (s == "--ppl-batch") a.ppl_batch = std::stoi(next());
        else if (s == "--kld-base") a.kld_base = next();
        else if (s == "--host") a.so.host = next();
        else if (s == "--port") a.so.port = std::stoi(next());
        else if (s == "--api-key") a.so.api_key = next();
        else if (s == "--alias") a.so.model_name = next();
        else if (s == "--raw") a.raw = true;
        else if (s == "--think") a.think = true;
        else if (s == "--temp") a.sp.temperature = std::stof(next());
        else if (s == "--top-k") a.sp.top_k = std::stoi(next());
        else if (s == "--top-p") a.sp.top_p = std::stof(next());
        else if (s == "--min-p") a.sp.min_p = std::stof(next());
        else if (s == "--presence-penalty") a.sp.presence_penalty = std::stof(next());
        else if (s == "--seed") a.sp.seed = std::stoull(next());
        else if (s == "-v") set_log_level(LogLevel::Debug);
        else { std::fprintf(stderr, "unknown option %s\n", s.c_str()); usage(); }
    }
    if (a.ec.model_path.empty()) usage();
    return a;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fatal("cannot read %s", path.c_str());
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string chat_prompt(const std::string& user, bool think) {
    std::string p = "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
    p += think ? "<think>\n" : "<think>\n\n</think>\n\n";
    return p;
}

void print_stats(const GenStats& s) {
    std::fprintf(stderr, "\nprompt: %d tokens in %.2f s (%.1f tok/s)\n", s.n_prompt, s.t_prompt_ms / 1000.0, s.prompt_tps());
    std::fprintf(stderr, "decode: %d tokens in %.2f s (%.2f tok/s)", s.n_gen, s.t_gen_ms / 1000.0, s.gen_tps());
    if (s.n_steps > 0)
        std::fprintf(stderr, ", %d steps, %.2f tokens/step, drafts accepted %d/%d (%.0f%%)", s.n_steps, (double) s.n_gen / s.n_steps,
                     s.n_accepted, s.n_drafted, s.n_drafted ? 100.0 * s.n_accepted / s.n_drafted : 0.0);
    std::fprintf(stderr, "\n");
    if (s.n_steps > 0)
        std::fprintf(stderr, "per step (ms): draft %.1f, verify %.1f, sample %.1f, rollback %.1f, mtp update %.1f\n",
                     s.t_draft_ms / s.n_steps, s.t_verify_ms / s.n_steps, s.t_sample_ms / s.n_steps, s.t_rollback_ms / s.n_steps,
                     s.t_mtp_ms / s.n_steps);
}

int cmd_run(Args& a) {
    Engine eng(a.ec);
    std::string text = a.file.empty() ? a.prompt : read_file(a.file);
    if (!a.raw) text = chat_prompt(text, a.think);
    const auto toks = eng.tokenizer().encode(text, true);
    Sampler smp(a.sp);
    std::string pending;
    auto st = eng.generate(toks, a.n_predict, smp, [&](int32_t t) {
        std::fputs(eng.tokenizer().token_to_piece(t, false).c_str(), stdout);
        std::fflush(stdout);
        return true;
    });
    print_stats(st);
    return 0;
}

int cmd_bench(Args& a) {
    Engine eng(a.ec);
    // deterministic prompt: a code-writing request padded with source text to -pp tokens
    std::string base = chat_prompt("Write a C function that parses a decimal integer from a string, handling sign and overflow. "
                                   "Explain it briefly afterwards.", false);
    auto toks = eng.tokenizer().encode(base, true);
    if ((int) toks.size() < a.pp) {
        auto filler = eng.tokenizer().encode(
            "/* The quick brown fox jumps over the lazy dog. 0123456789 */\nint f(int x) { return x * 31 + 7; }\n", false);
        std::vector<int32_t> pre;
        while ((int) (pre.size() + toks.size()) < a.pp) pre.insert(pre.end(), filler.begin(), filler.end());
        pre.resize(a.pp - toks.size());
        toks.insert(toks.begin(), pre.begin(), pre.end());
    }
    Sampler smp(a.sp);
    auto st = eng.generate(toks, a.n_predict, smp, [](int32_t) { return true; });
    print_stats(st);
    std::printf("pp %d: %.1f tok/s | tg %d: %.2f tok/s | VRAM free after run %.0f MiB\n", st.n_prompt, st.prompt_tps(), st.n_gen,
                st.gen_tps(), eng.vram_free() / MiB);
    return 0;
}

// llama-perplexity's --kl-divergence-base file: "_logits_", n_ctx, n_vocab, n_chunk, tokens[n_chunk * n_ctx], then
// per chunk (n_ctx - 1 - n_ctx/2) rows of nv uint16 (nv = 2*((n_vocab+1)/2) + 4): two floats (scale, min_log_prob)
// followed by quantized log-probabilities.
struct KldBase {
    int n_ctx = 0, n_vocab = 0, n_chunk = 0, nv = 0;
    std::vector<int32_t> tokens;
    std::ifstream f;
    std::vector<uint16_t> row;
    bool open(const std::string& path) {
        f.open(path, std::ios::binary);
        char magic[8];
        if (!f.read(magic, 8) || std::memcmp(magic, "_logits_", 8) != 0) return false;
        f.read((char*) &n_ctx, 4);
        f.read((char*) &n_vocab, 4);
        f.read((char*) &n_chunk, 4);
        tokens.resize((size_t) n_chunk * n_ctx);
        f.read((char*) tokens.data(), tokens.size() * 4);
        nv = 2 * ((n_vocab + 1) / 2) + 4;
        row.resize(nv);
        return (bool) f;
    }
    // next base row as log-probabilities
    void next(std::vector<float>& lp) {
        f.read((char*) row.data(), (size_t) nv * 2);
        float scale, minlp;
        std::memcpy(&scale, row.data(), 4);
        std::memcpy(&minlp, row.data() + 2, 4);
        lp.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) lp[i] = scale * row[4 + i] + minlp;
    }
};

int cmd_ppl(Args& a) {
    a.ec.n_draft = 0;  // no MTP needed
    KldBase base;
    const bool kld = !a.kld_base.empty();
    if (kld && !base.open(a.kld_base)) fatal("cannot read KL-divergence base '%s'", a.kld_base.c_str());
    const int c = kld ? base.n_ctx : a.ppl_ctx;
    a.ec.n_ctx = c;
    a.ec.n_ubatch = std::max(a.ec.n_ubatch, a.ppl_batch);
    Engine eng(a.ec);
    const auto toks = kld ? base.tokens : eng.tokenizer().encode(read_file(a.file), false);
    int n_chunks = (int) toks.size() / c;
    if (a.ppl_chunks > 0) n_chunks = std::min(n_chunks, a.ppl_chunks);
    if (n_chunks < 1) fatal("text too short: %zu tokens for context %d", toks.size(), c);
    const int n_vocab = eng.n_vocab();
    if (kld && base.n_vocab != n_vocab) fatal("vocabulary size mismatch with the base file");
    const int first = c / 2;
    const int sub = a.ppl_batch;  // tokens per pass with all logits (bounds the logits buffer)
    double nll = 0.0, nll_base = 0.0, kl_sum = 0.0;
    long count = 0, same_top = 0;
    std::vector<float> logits, blp;
    const double t0 = now_ms();
    for (int ch = 0; ch < n_chunks; ++ch) {
        eng.reset();
        const int32_t* t = toks.data() + (size_t) ch * c;
        for (int i = 0; i < c; i += sub) {
            const int nb = std::min(sub, c - i);
            eng.eval(t + i, nb, nb, &logits);
            for (int j = 0; j < nb; ++j) {
                const int pos = i + j;
                if (pos < first || pos + 1 >= c) continue;
                const float* l = logits.data() + (size_t) j * n_vocab;
                float mx = l[0];
                int imax = 0;
                for (int v = 1; v < n_vocab; ++v)
                    if (l[v] > mx) { mx = l[v]; imax = v; }
                double sum = 0.0;
                for (int v = 0; v < n_vocab; ++v) sum += std::exp((double) l[v] - mx);
                const double lse = mx + std::log(sum);
                nll += lse - l[t[pos + 1]];
                ++count;
                if (kld) {
                    base.next(blp);
                    nll_base += -blp[t[pos + 1]];
                    double kl = 0.0;
                    int ibase = 0;
                    for (int v = 0; v < n_vocab; ++v) {
                        if (blp[v] > blp[ibase]) ibase = v;
                        if (blp[v] > -16.0f) kl += std::exp((double) blp[v]) * (blp[v] - (l[v] - lse));
                    }
                    kl_sum += kl;
                    if (ibase == imax) ++same_top;
                }
            }
        }
        if (kld)
            std::printf("[%d] PPL %.4f  base %.4f  mean KLD %.5f  same top %.2f%%\n", ch + 1, std::exp(nll / count), std::exp(nll_base / count),
                        kl_sum / count, 100.0 * same_top / count);
        else
            std::printf("[%d] %.4f\n", ch + 1, std::exp(nll / count));
        std::fflush(stdout);
    }
    std::printf("Final estimate: PPL = %.4f (%ld tokens scored, %.1f s)\n", std::exp(nll / count), count, (now_ms() - t0) / 1000.0);
    if (kld)
        std::printf("vs base: base PPL = %.4f, mean KLD = %.5f, same top token %.2f%%\n", std::exp(nll_base / count), kl_sum / count,
                    100.0 * same_top / count);
    return 0;
}

// Speculative decoding must emit exactly what the model would emit one token at a time (at temperature 0), up to
// numerical noise between batch shapes. Generate with drafts, then replay the tokens without speculation and
// report every position where the emitted token is not the plain model's top choice, with the logit margin.
int cmd_check_spec(Args& a) {
    Engine eng(a.ec);
    std::string text = a.file.empty() ? a.prompt : read_file(a.file);
    if (!a.raw) text = chat_prompt(text, a.think);
    const auto prompt = eng.tokenizer().encode(text, true);
    SamplerParams sp;
    sp.temperature = 0.0f;
    Sampler smp(sp);
    std::vector<int32_t> gen;
    const auto st = eng.generate(prompt, a.n_predict, smp, [&](int32_t t) {
        gen.push_back(t);
        return true;
    });
    std::fprintf(stderr, "generated %zu tokens with drafts (%d steps, %d/%d drafts accepted)\n", gen.size(), st.n_steps, st.n_accepted, st.n_drafted);

    eng.reset();
    std::vector<float> logits;
    eng.eval(prompt.data(), (int) prompt.size(), 1, &logits);
    const int nv = eng.n_vocab();
    int mism = 0, bad = 0;
    double worst = 0.0;
    for (size_t i = 0; i < gen.size(); ++i) {
        const float* l = logits.data();
        const int top = (int) (std::max_element(l, l + nv) - l);
        if (top != gen[i]) {
            const double margin = (double) l[top] - l[gen[i]];
            ++mism;
            worst = std::max(worst, margin);
            if (margin > 0.5) ++bad;
            std::printf("pos %zu: emitted %d, plain top %d, margin %.4f\n", i, gen[i], top, margin);
        }
        eng.eval(&gen[i], 1, 1, &logits);
    }
    std::printf("check-spec: %zu tokens, %d differ from the plain top-1 (worst margin %.4f), %d with margin > 0.5 -> %s\n", gen.size(),
                mism, worst, bad, bad == 0 ? "OK (numerical near-ties only)" : "SUSPICIOUS");
    return bad == 0 ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    if (a.cmd == "run") return cmd_run(a);
    if (a.cmd == "bench") return cmd_bench(a);
    if (a.cmd == "ppl") return cmd_ppl(a);
    if (a.cmd == "check-spec") return cmd_check_spec(a);
    if (a.cmd == "serve") {
        Engine eng(a.ec);
        return run_server(eng, a.so);
    }
    usage();
}

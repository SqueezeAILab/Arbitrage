# Arbitrage

Modern large language models achieve impressive reasoning capabilities with long chains of thought, but they incur substantial computational cost at inference time. Speculative decoding improves efficiency by using a fast, less accurate draft model to propose tokens that are then verified in parallel by a stronger target model. However, on reasoning tasks, traditional token-level speculative decoding often rejects many semantically valid steps due to superficial token mismatches. Recent step-level semantic verification methods mitigate this by accepting or rejecting entire reasoning steps, but they still waste target compute by regenerating many rejected steps that yield little quality gain.

We propose **ARBITRAGE**, a step-level speculative generation framework that dynamically routes generation based on the *relative* advantage of the target model over the draft model. Instead of relying on a fixed acceptance threshold, ARBITRAGE uses a lightweight router trained to predict when the target model is likely to produce a meaningfully better step. This routing approximates an ideal “arbitrage oracle” that always selects the higher-quality step, achieving near-optimal efficiency–accuracy trade-offs. Across multiple mathematical reasoning benchmarks, ARBITRAGE consistently outperforms prior step-level speculative decoding baselines, reducing inference latency by up to approximately 2× at matched accuracy.

<p float="left" align="middle">
  <img src="./images/method.pdf" width="750">
</p>


## What’s here
- `arbitrage.py`: main entrypoint.
- `scripts/serve_*.sh`: helpers to launch draft/target/PRM/router OpenAI-style endpoints (sglang or vLLM).
- `scripts/example_run.sh`: sample sweep over PRM thresholds.

## Prerequisites
- Python 3.10+ with CUDA GPUs for vLLM/sglang.
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  # sglang is used for draft/target servers if you follow the scripts
  pip install "sglang"
  # For using Skywork-PRM
  git clone https://github.com/SkyworkAI/skywork-o1-prm-inference.git
  cd skywork-o1-prm-inference
  pip install -e .
  # For inference on the trained router
  cd utils/router_inference
  pip install -e .
  
  ```

## Start model servers (example defaults)
Each script prints the host IP and exposes an OpenAI-compatible endpoint.
- Draft (fast): `bash scripts/serve_draft.sh` (or `serve_draft_quantized.sh` for GGUF).
- Target (accurate): `bash scripts/serve_target.sh`.
- PRM scorer: `bash scripts/serve_prm.sh`.
- Router model (optional, for `run_type=router`): `bash scripts/serve_router.sh`.

## Acknowledgement
Our code base mainly builds on [Reward-Guided Speculative Decoding (RSD)](https://github.com/BaohaoLiao/RSD), [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math), and [skywork-o1-prm-inference](https://github.com/SkyworkAI/skywork-o1-prm-inference).

## Citation
```
@misc{maheswaran2025arbitrageefficientreasoningadvantageaware,
      title={Arbitrage: Efficient Reasoning via Advantage-Aware Speculation}, 
      author={Monishwaran Maheswaran and Rishabh Tiwari and Yuezhou Hu and Kerem Dilmen and Coleman Hooper and Haocheng Xi and Nicholas Lee and Mehrdad Farajtabar and Michael W. Mahoney and Kurt Keutzer and Amir Gholami},
      year={2025},
      eprint={2512.05033},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.05033}, 
}
```
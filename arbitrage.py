# THIS CODE IS ADAPTED FROM THE RSD CODE
# https://github.com/BaohaoLiao/RSD
# https://arxiv.org/pdf/2501.19324?

import random
import os
import argparse
import time
import json
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer
from openai import OpenAI

from utils.qwen25_math_evaluation.evaluate import evaluate
from utils.qwen25_math_evaluation.utils import (
    set_seed,
    load_jsonl,
    save_jsonl,
    construct_prompt,
)
from utils.qwen25_math_evaluation.parser import *
from utils.qwen25_math_evaluation.trajectory import *
from utils.qwen25_math_evaluation.data_loader import load_data
from utils.qwen25_math_evaluation.python_executor import PythonExecutor
from utils.skywork_o1_prm_inference.model_utils.io_utils import (
    prepare_input,
    derive_step_rewards_vllm,
    derive_step_rewards_vllm_router,
)

# For parsing arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="math500", type=str)
    # Here the run type can be rsd, generate, oracle, router
    parser.add_argument(
        "--run_type", default="rsd", type=str
    ) 
    parser.add_argument(
        "--data_dir", default="./utils/qwen25_math_evaluation/data", type=str
    )
    parser.add_argument(
        "--draft_model_name_or_path",
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        type=str,
    )
    parser.add_argument(
        "--draft_model_ip_address", default="http://localhost:12340/v1", type=str
    )
    parser.add_argument(
        "--draft_model_tokenizer", default="Qwen/Qwen2.5-Math-1.5B-Instruct", type=str
    )
    parser.add_argument(
        "--target_model_name_or_path", default="Qwen/Qwen2.5-Math-7B-Instruct", type=str
    )
    parser.add_argument(
        "--target_model_ip_address", default="http://localhost:12341/v1", type=str
    )
    parser.add_argument(
        "--prm_name_or_path",
        default="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
        type=str,
    )
    parser.add_argument(
        "--prm_ip_address", default="http://localhost:12342/v1", type=str
    )
    parser.add_argument(
        "--router_name_or_path",
        default="HuggingFaceH4/Qwen2.5-Math-1.5B-Instruct-PRM-0.2",
        type=str,
    )
    parser.add_argument(
        "--router_ip_address",
        default="http://localhost:12343/v1",
        type=str,
    )
    parser.add_argument(
        "--router_tokenizer",
        default="HuggingFaceH4/Qwen2.5-Math-1.5B-Instruct-PRM-0.2",
        type=str,
    )

    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--timeout", default=100000000, type=int)
    parser.add_argument("--prompt_type", default="qwen25-math-cot", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--step_word", type=str, default="\n\n")
    parser.add_argument("--step_word_newline", action="store_true")
    parser.add_argument("--prm_threshold", type=float, default=0.7)
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--ordinal", action="store_true")
    parser.add_argument("--classification_4", action="store_true")
    parser.add_argument("--generate_tag", type=str, default="nw0902")
    parser.add_argument("--multi_target_run", action="store_true")
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    parser.add_argument("--annotate", action="store_true")
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)

    if args.step_word_newline:
        args.step_word = "\n"
    else:
        args.step_word = "\n\n"

    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)
    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]
    # get out_file name
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}_delta{args.prm_threshold}_maxsteps{args.max_steps}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(list(load_jsonl(f"{output_dir}/{data_name}/{f}")))

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def setup(args):
    # load model
    openai_api_key = "EMPTY"
    draft_client = OpenAI(
        api_key=openai_api_key,
        base_url=args.draft_model_ip_address,
        timeout=args.timeout,
    )
    draft_tokenizer = AutoTokenizer.from_pretrained(
        args.draft_model_tokenizer,
        trust_remote_code=True,
    )

    target_client = OpenAI(
        api_key=openai_api_key,
        base_url=args.target_model_ip_address,
        timeout=args.timeout,
    )
    target_tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_name_or_path, trust_remote_code=True
    )

    prm_client = None
    prm_tokenizer = None
    router_client = None
    router_tokenizer = None

    if args.run_type != "router":
        prm_client = OpenAI(
            api_key=openai_api_key,
            base_url=args.prm_ip_address,
            timeout=args.timeout,
        )
        prm_tokenizer = AutoTokenizer.from_pretrained(
            args.prm_name_or_path, trust_remote_code=True
        )
    else:
        router_client = OpenAI(
            api_key=openai_api_key,
            base_url=args.router_ip_address,
            timeout=args.timeout,
        )
        router_tokenizer = AutoTokenizer.from_pretrained(
            args.router_tokenizer,
            trust_remote_code=True,
            use_fast=True,
        )

    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        results.append(
            main(
                draft_client,
                target_client,
                prm_client,
                router_client,
                draft_tokenizer,
                target_tokenizer,
                prm_tokenizer,
                router_tokenizer,
                data_name,
                args,
            )
        )

    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["acc"] for result in results]) / len(results),
        }
    )

    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True

def prepare_input_router(
    problem,
    response,
    tokenizer,
    step_token,
):
    prompt_ids = tokenizer.encode(problem + "\n")
    response_ids = []
    steps = []
    reward_flags = [0] * len(prompt_ids)
    step_token_id = tokenizer.encode(step_token)[-1]
    all_responses = response.split(step_token)
    for idx, step in enumerate(all_responses):
        if step != "":
            step_ids = tokenizer.encode(step)
        else:
            step_ids = []
        step_ids += [step_token_id]
        step = step + step_token
        flag = [0] * len(step_ids)
        flag[-1] = 1
        response_ids.extend(step_ids)
        reward_flags.extend(flag)
        steps.append(step)
    return prompt_ids + response_ids, steps, reward_flags


def generate_with_model(
    args,   
    model_name,
    prm_model_name,
    temperature,
    top_p,
    max_tokens_per_call,
    step_word,
    client,
    prm_client,
    prm_tokenizer,
    batch_prompts,
    current_prompts,
    current_problems,
    annotatation="",
    router_model_name=None,
    router_client=None,
    router_tokenizer=None,
):
    # First generate with the model
    responses = client.completions.create(
        model=model_name,
        prompt=batch_prompts,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens_per_call,
        stop=[step_word],
    ).choices
    responses = sorted(responses, key=lambda x: int(x.index))

    full_responses = [
        "".join(r[0] for r in prev_resp) + new_resp.text
        for (_, _, prev_resp), new_resp in zip(current_prompts, responses)
    ]

    full_responses_annotated = [
        "".join("Model " + str(r[1] - 1) + ": " + r[0] for r in prev_resp)
        + annotatation
        + new_resp.text
        for (_, _, prev_resp), new_resp in zip(current_prompts, responses)
    ]

    if args.run_type != "router":
        processed_data = [
            prepare_input(p, full_resp, tokenizer=prm_tokenizer, step_token=step_word)
            for p, full_resp in zip(current_problems, full_responses)
        ]

        input_ids, steps, reward_flags = zip(*processed_data)
        rewards = prm_client.embeddings.create(
            input=input_ids,
            model=prm_model_name,
        )
        step_rewards = derive_step_rewards_vllm(rewards, reward_flags)  # list[list]
    else:
        step_rewards = [None] * len(responses)

    if args.run_type == "router" and router_client is not None:
        processed_data_router = [
                prepare_input_router(
                    p, full_resp, tokenizer=router_tokenizer, step_token=args.step_word
                )
                for p, full_resp in zip(
                    current_problems,
                    full_responses_annotated if args.annotate else full_responses,
                )
            ]
        input_ids_router, _, _ = zip(*processed_data_router)
        router_rewards = router_client.embeddings.create(
            input=input_ids_router,
            model=router_model_name,
        )
        # take the last two values of the router_rewards
        router_outputs = derive_step_rewards_vllm_router(router_rewards)
    else:
        router_outputs = [None] * len(responses)

    return responses, step_rewards, full_responses, router_outputs, full_responses_annotated

def vector_sum(rewards, target_step_rewards):
    outputs = []
    for r, tr in zip(rewards, target_step_rewards):
        output = [0] * len(r)
        for i in range(len(r)):
            output[i] = r[i] + tr[i]
        outputs.append(output)
    return outputs


def get_responses(
    args,
    draft_client,
    target_client,
    prm_client,
    router_client,
    draft_tokenizer,
    target_tokenizer,
    prm_tokenizer,
    router_tokenizer,
    prompts,
    problems,
):
    outputs = [None] * len(prompts)  # Initialize with None for tracking
    token_counts = [
        (0, 0, 0) for _ in prompts
    ]  # (draft_tokens, target_tokens, discarded_draft_tokens) for each prompt
    step_info = [
        [] for _ in prompts
    ]  # List to store (step_num, client_id) for each prompt
    current_prompts = [
        (i, p, []) for i, p in enumerate(prompts)
    ]  # (index, prompt, responses)
    all_rewards = [
        [] for _ in prompts
    ]  # List to store (step_num, client_id) for each prompt
    current_problems = problems
    num_step = 0
    pre_num_finished = 0
    num_unchanged = 0

    data_to_dump = []

    while current_prompts:
        batch_prompts = [
            p + "".join(r[0] for r in responses) for _, p, responses in current_prompts
        ]

        (
            draft_responses,
            step_rewards,
            full_responses_draft,
            router_outputs_draft,
            full_responses_annotated_draft,
        ) = generate_with_model(
            args,
            model_name=args.draft_model_name_or_path.split("/")[-1],
            prm_model_name=args.prm_name_or_path.split("/")[-1],
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens_per_call=args.max_tokens_per_call,
            step_word=args.step_word,
            client=draft_client,
            prm_client=prm_client,
            prm_tokenizer=prm_tokenizer,
            batch_prompts=batch_prompts,
            current_prompts=current_prompts,
            current_problems=current_problems,
            router_model_name=args.router_name_or_path.split("/")[-1],
            router_client=router_client,
            router_tokenizer=router_tokenizer,
            annotatation="Model 0: ",
        )

        (
            target_responses,
            target_step_rewards,
            full_responses_target,
            _,
            full_responses_annotated_target,
        ) = generate_with_model(
            args,
            model_name=args.target_model_name_or_path.split("/")[-1],
            prm_model_name=args.prm_name_or_path.split("/")[-1],
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens_per_call=args.max_tokens_per_call,
            step_word=args.step_word,
            client=target_client,
            prm_client=prm_client,
            prm_tokenizer=prm_tokenizer,
            batch_prompts=batch_prompts,
            current_prompts=current_prompts,
            current_problems=current_problems,
            annotatation="Model 1: ",
        )

        if args.multi_target_run:
            for _ in range(3):
                target_step_rewards = vector_sum(
                    target_step_rewards,
                    generate_with_model(
                        args,
                        model_name=args.target_model_name_or_path.split("/")[-1],
                        prm_model_name=args.prm_name_or_path.split("/")[-1],
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens_per_call=args.max_tokens_per_call,
                        step_word=args.step_word,
                        client=target_client,
                        prm_client=prm_client,
                        prm_tokenizer=prm_tokenizer,
                        batch_prompts=batch_prompts,
                        current_prompts=current_prompts,
                        current_problems=current_problems,
                        annotatation="Model 1: ",
                    )[1],
                )
            target_step_rewards = [[x / 4 for x in r] for r in target_step_rewards]

        if args.run_type == "generate":
            for (
                (orig_idx, prompt, prev_responses),
                problem,
                draft_response,
                target_response,
                step_reward,
                target_step_reward,
                full_response_draft,
                full_response_target,
                full_response_annotated_draft,
                full_response_annotated_target,
            ) in zip(
                current_prompts,
                current_problems,
                draft_responses,
                target_responses,
                step_rewards,
                target_step_rewards,
                full_responses_draft,
                full_responses_target,
                full_responses_annotated_draft,
                full_responses_annotated_target,
            ):
                data_to_dump.append(
                    {
                        "idx": orig_idx,
                        "problem": problem,
                        "prompt": prompt,
                        "prev_responses": prev_responses,
                        "draft_response": draft_response.text,
                        "target_response": target_response.text,
                        "step_reward": step_reward,
                        "target_step_reward": target_step_reward,
                        "full_response_draft": full_response_draft,
                        "full_response_target": full_response_target,
                        "full_response_annotated_draft": full_response_annotated_draft,
                        "full_response_annotated_target": full_response_annotated_target,
                    }
                )

        good_prompts = []
        for (
            (orig_idx, prompt, prev_responses),
            draft_response,
            target_response,
            step_reward,
            target_step_reward,
            router_output,
        ) in zip(
            current_prompts,
            draft_responses,
            target_responses,
            step_rewards,
            target_step_rewards,
            router_outputs_draft,
        ):
            if args.run_type != "router":
                all_rewards[orig_idx].append(round(step_reward[-1], 6))
                delta = target_step_reward[-1] - step_reward[-1]
                differ = delta > args.prm_threshold

            if args.run_type == "rsd":
                if step_reward[-1] >= args.prm_threshold:
                    good_prompts.append(
                        (orig_idx, prompt, prev_responses, draft_response, True)
                    )  # True means using draft model
                else:
                    draft_response_text = draft_response.text + args.step_word
                    token_counts[orig_idx] = (
                        token_counts[orig_idx][0],
                        token_counts[orig_idx][1],
                        token_counts[orig_idx][2]
                        + len(draft_tokenizer.encode(draft_response_text)),
                    )
                    good_prompts.append(
                        (orig_idx, prompt, prev_responses, target_response, False)
                    )
                    # bad_prompts.append((orig_idx, prompt, prev_responses))
            elif args.run_type == "oracle":
                if not differ:
                    good_prompts.append(
                        (orig_idx, prompt, prev_responses, draft_response, True)
                    )  # True means using draft model
                else:
                    draft_response_text = draft_response.text + args.step_word
                    token_counts[orig_idx] = (
                        token_counts[orig_idx][0],
                        token_counts[orig_idx][1],
                        token_counts[orig_idx][2]
                        + len(draft_tokenizer.encode(draft_response_text)),
                    )
                    good_prompts.append(
                        (orig_idx, prompt, prev_responses, target_response, False)
                    )
            elif args.run_type == "generate":
                if step_reward[-1] >= target_step_reward[-1]:
                    good_prompts.append(
                        (orig_idx, prompt, prev_responses, draft_response, True)
                    )  # True means using draft model
                else:
                    draft_response_text = draft_response.text + args.step_word
                    token_counts[orig_idx] = (
                        token_counts[orig_idx][0],
                        token_counts[orig_idx][1],
                        token_counts[orig_idx][2]
                        + len(draft_tokenizer.encode(draft_response_text)),
                    )
                    good_prompts.append(
                        (orig_idx, prompt, prev_responses, target_response, False)
                    )
            elif args.run_type == "router":
                router_decision = router_output >= args.prm_threshold
                if not router_decision:
                    good_prompts.append(
                        (orig_idx, prompt, prev_responses, draft_response, True)
                    )  # True means using draft model
                else:
                    draft_response_text = draft_response.text + args.step_word
                    token_counts[orig_idx] = (
                        token_counts[orig_idx][0],
                        token_counts[orig_idx][1],
                        token_counts[orig_idx][2]
                        + len(draft_tokenizer.encode(draft_response_text)),
                    )
                    good_prompts.append(
                        (orig_idx, prompt, prev_responses, target_response, False)
                    )
            else:
                raise ValueError(f"Invalid run type: {args.run_type}")

        # Process all responses
        next_prompts = []
        next_problems = []
        for orig_idx, prompt, prev_responses, response, used_draft in sorted(
            good_prompts, key=lambda x: x[0]
        ):
            response_text = response.text + args.step_word
            client_id = 1 if used_draft else 2
            tokenizer = draft_tokenizer if client_id == 1 else target_tokenizer
            num_tokens = len(tokenizer.encode(response_text))

            # Update token counts
            if client_id == 1:
                token_counts[orig_idx] = (
                    token_counts[orig_idx][0] + num_tokens,
                    token_counts[orig_idx][1],
                    token_counts[orig_idx][2],
                )
            else:
                token_counts[orig_idx] = (
                    token_counts[orig_idx][0],
                    token_counts[orig_idx][1] + num_tokens,
                    token_counts[orig_idx][2],
                )

            # Record step information
            step_info[orig_idx].append((num_step, client_id))

            full_responses = prev_responses + [(response_text, client_id)]
            full_responses_text = "".join(r[0] for r in full_responses)

            # terminate conditions
            if (
                (response.matched_stop != args.step_word)
                or len(draft_tokenizer.encode(prompt + full_responses_text))
                >= args.max_tokens_per_call
                or len(target_tokenizer.encode(prompt + full_responses_text))
                >= args.max_tokens_per_call
                or num_step >= args.max_steps - 1
                or num_unchanged >= args.patience - 1
            ):
                outputs[orig_idx] = full_responses_text[: -len(args.step_word)]
            else:
                next_prompts.append((orig_idx, prompt, full_responses))
                next_problems.append(problems[orig_idx])

        current_prompts = next_prompts
        current_problems = next_problems
        assert len(current_prompts) == len(current_problems)
        if len(outputs) - len(current_prompts) > pre_num_finished:
            num_unchanged = 0
            pre_num_finished = len(outputs) - len(current_prompts)
        else:
            num_unchanged += 1

        print(
            f"#### Step {num_step}: Completed {pre_num_finished} / {len(outputs)}, #unchanged {num_unchanged} / {args.patience}"
        )
        num_step += 1
        
        # dump the data to a jsonl file for every iteration in case the loop is interrupted
        if args.run_type == "generate":
            save_jsonl(
                data_to_dump,
                f"dumps/{args.data_names}/{args.generate_tag}_{args.data_names}_{args.run_type}_{args.prm_name_or_path.split('/')[-1]}_draft_{args.draft_model_name_or_path.split('/')[-1]}_target_{args.target_model_name_or_path.split('/')[-1]}_{args.temperature}_{args.top_p}_start_{args.start}_end_{args.end}.jsonl",
            )

    # dump the data to a jsonl file for the final iteration
    if args.run_type == "generate":
        save_jsonl(
            data_to_dump,
            f"dumps/{args.data_names}/{args.generate_tag}_{args.data_names}_{args.run_type}_{args.prm_name_or_path.split('/')[-1]}_draft_{args.draft_model_name_or_path.split('/')[-1]}_target_{args.target_model_name_or_path.split('/')[-1]}_{args.temperature}_{args.top_p}_start_{args.start}_end_{args.end}.jsonl",
        )

    return outputs, token_counts, step_info, all_rewards


def main(
    draft_client,
    target_client,
    prm_client,
    router_client,
    draft_tokenizer,
    target_tokenizer,
    prm_tokenizer,
    router_tokenizer,
    data_name,
    args,
):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    if args.apply_chat_template:
        input_prompts = [
            draft_tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")

    # start inference
    start_time = time.time()
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]
        problems = [sample["question"] for sample in samples]
        assert len(prompts) == len(problems)
        outputs, token_counts, turn_info, all_rewards = get_responses(
            args,
            draft_client,
            target_client,
            prm_client,
            router_client,
            draft_tokenizer,
            target_tokenizer,
            prm_tokenizer,
            router_tokenizer,
            prompts,
            problems,
        )
        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif "boxed" not in output and output.endswith("```"):
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # execute the remain prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # extract preds
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.pop("prompt")
        sample.update(
            {
                "code": code,
                "pred": preds,
                "report": reports,
                "token_counts": token_counts[i],
                "turn_info": turn_info[i],
                "reward": all_rewards[i],
            }
        )
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)

    # save metrics
    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )

    llm1_tokens = [0, 0]  # (correct, wrong)
    llm1_discarded_tokens = [0, 0]
    llm2_tokens = [0, 0]
    for i, sample in enumerate(all_samples):
        if sample["score"][0]:
            llm1_tokens[0] += sample["token_counts"][0]
            llm2_tokens[0] += sample["token_counts"][1]
            llm1_discarded_tokens[0] += sample["token_counts"][2]
        else:
            llm1_tokens[1] += sample["token_counts"][0]
            llm2_tokens[1] += sample["token_counts"][1]
            llm1_discarded_tokens[1] += sample["token_counts"][2]
    total_tokens = sum(llm1_tokens) + sum(llm2_tokens) + sum(llm1_discarded_tokens)
    total_tokens_for_correct_pred = (
        llm1_discarded_tokens[0] + llm1_tokens[0] + llm2_tokens[0]
    )
    total_tokens_for_wrong_pred = (
        llm1_discarded_tokens[1] + llm1_tokens[1] + llm2_tokens[1]
    )

    result_json["tokens_ratio_overall(llm1,llm2)"] = (
        (
            (sum(llm1_tokens) + sum(llm1_discarded_tokens)) / total_tokens,
            sum(llm2_tokens) / total_tokens,
        )
        if total_tokens > 0
        else (0, 0)
    )
    result_json["tokens_ratio_correct_prediction(llm1,llm2)"] = (
        (
            (llm1_discarded_tokens[0] + llm1_tokens[0]) / total_tokens_for_correct_pred,
            llm2_tokens[0] / total_tokens_for_correct_pred,
        )
        if total_tokens_for_correct_pred > 0
        else (0, 0)
    )
    result_json["tokens_ratio_wrong_prediction(llm1,llm2)"] = (
        (
            (llm1_discarded_tokens[1] + llm1_tokens[1]) / total_tokens_for_wrong_pred,
            llm2_tokens[1] / total_tokens_for_wrong_pred,
        )
        if total_tokens_for_wrong_pred > 0
        else (0, 0)
    )
    result_json["tokens_ratio(correct,wrong)"] = (
        (
            total_tokens_for_correct_pred / total_tokens,
            total_tokens_for_wrong_pred / total_tokens,
        )
        if total_tokens > 0
        else (0, 0)
    )
    result_json["tokens_ratio_discarded(correct,wrong)"] = (
        (
            llm1_discarded_tokens[0] / total_tokens_for_correct_pred,
            llm1_discarded_tokens[1] / total_tokens_for_wrong_pred,
        )
        if (total_tokens_for_correct_pred > 0 and total_tokens_for_wrong_pred > 0)
        else (0, 0)
    )
    result_json["acceptance_rate"] = (
        (
            (llm1_tokens[0] + llm1_tokens[1])
            / (
                llm1_tokens[0]
                + llm1_tokens[1]
                + llm1_discarded_tokens[0]
                + llm1_discarded_tokens[1]
            )
        )
        if ((llm1_tokens[0] + llm1_tokens[1]) > 0)
        else 0
    )
    result_json["num_draft_tokens"] = sum(llm1_tokens) + sum(llm1_discarded_tokens)
    result_json["num_target_tokens"] = sum(llm2_tokens)

    with open(
        out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    return result_json


if __name__ == "__main__":
    args = parse_args()
    print(args)
    set_seed(args.seed)
    setup(args)

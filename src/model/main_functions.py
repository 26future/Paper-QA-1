import os
import torch
import timeit

from fastprogress.fastprogress import master_bar, progress_bar
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW,get_linear_schedule_with_warmup

from src.functions.mrc_metrics import compute_predictions_logits, normalize_answer
from src.functions.processor_plus import SquadResult
from src.functions.utils import load_and_cache_dataset, set_seed, to_list
from src.functions.evaluate import squad_evaluate


def train(args, model, tokenizer, logger):

    global_step = 1
    tr_loss = 0.0
    model.zero_grad()

    mb = master_bar(range(int(args.num_train_epochs)))
    set_seed(args)

    for epoch in mb:
        for version in range(args.dataset_nums):
            train_dataset = load_and_cache_dataset(args, tokenizer, evaluate=False, output_examples=False,dataset_num=version)

            logger.info(f" ***** Loading Dataset ***** train : {version + 1}/{args.dataset_nums}")
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

            # Layer에 따른 가중치 decay 적용
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
                },
                {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0},
            ]

            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
            )

            # training info
            logger.info(f"***** Running training : train_{version}*****")
            logger.info("  Num examples = %d", len(train_dataset))
            logger.info("  Num Epochs = %d", args.num_train_epochs)
            logger.info("  Train batch size per GPU = %d", args.train_batch_size)
            logger.info(
                "  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size
                * args.gradient_accumulation_steps)
            logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
            logger.info("  Total optimization steps = %d", t_total)

            # 학습 시작
            epoch_iterator = progress_bar(train_dataloader, parent=mb)
            for step, batch in enumerate(epoch_iterator):
                model.train()
                batch = tuple(t.to(args.device) for t in batch)

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "start_positions": batch[3],
                    "end_positions": batch[4],
                    "keyword_labels": batch[7],
                    "evidence_labels": batch[8],
                }

                del inputs["token_type_ids"]     # roberta

                outputs = model(**inputs)
                loss = outputs[0]

                # gradient accumulation
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()

                if (global_step + 1) % 100 == 0:
                    print(
                        "{} step processed.. Current Total Loss : {}\n".format(
                            (global_step + 1), loss.item(),
                        ))

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    # model save
                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:

                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        # 저장
                        model.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Validation Test!!
                        logger.info("***** Eval results *****")
                        results = evaluate(args, model, tokenizer, logger, global_step=global_step)

        #######################
        # Epoch 마다 저장
        #######################
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)
        mb.write("Epoch {} done".format(epoch + 1))

        # Validation Test!!
        logger.info(f"***** {epoch+1} Eval results *****")
        results = evaluate(args, model, tokenizer, logger, global_step=global_step)

    return global_step, tr_loss / global_step


def evaluate(args, model,tokenizer, logger, global_step = ""):

    dataset, examples, features = load_and_cache_dataset(args, tokenizer, evaluate=True, output_examples=True)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(global_step))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    logger.info("  Evaluated File = {}".format(args.predict_file))

    all_results = []
    start_time = timeit.default_timer()

    # output directory 설정
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_dir = args.output_dir

    for batch in progress_bar(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "keyword_labels": batch[6],
                "evidence_labels": batch[7]
            }

            del inputs["token_type_ids"]

            example_indices = batch[3]

            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):

            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            # outputs = [start_logits, end_logits]
            output = [to_list(output[i]) for output in outputs]

            # start_logits: [batch_size, max_length]
            # end_logits: [batch_size, max_length]
            start_logits, end_logits = output

            result = SquadResult(unique_id, start_logits, end_logits)
            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    devfile_name = str(args.predict_file).split("/")[-1].split(".")[0]  # validation 파일 명
    output_prediction_file = os.path.join(args.output_dir, "qa_predictions_{}.json".format(global_step if global_step is not "" else devfile_name))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(global_step if global_step is not "" else devfile_name))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(global_step if global_step is not "" else devfile_name))
        logger.info("  Evaluation - with version2 with negative -")
    else:
        output_null_log_odds_file = None
        logger.info("  Evaluation - without version2 -")

    # 각 result 값 저장
    torch.save(all_results, os.path.join(output_dir,"all_results"))

    # 결과 예측
    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )

    final_results = squad_evaluate(examples, predictions)

    for key in sorted(final_results.keys()):
        logger.info("  %s = %s", key, str(final_results[key]))

    return final_results


##############################################
#                  데모용 코드               #
##############################################

# -*- coding: utf-8 -*-
from src.functions.utils import load_examples

def predict(args, model, tokenizer, logger):
    while(True):
        # 질문에 대한 정답을 찾기위한 문서 입력
        context = input("\n Paper Context : ")
        if context == '-1':
            exit(1)
        question = input("\n Question about the paper : ")
        # 검색 모델 결과
        keyword = input("\n Question Keyword from Keyword Model: ")
        evidence = input("\n Evidence Sentence from IR Model : ")

        input_context = context   # normalize_answer(context)
        input_question = question    # normalize_answer(question)
        input_keyword = keyword    # normalize_answer(keyword)
        input_evidence = evidence    # normalize_answer(evidence)

        if input_context.strip() == "" or input_question.strip() == "":
            logger.info("input_context : " + str(input_context) + "\n")
            logger.info("input_question : " + str(input_question) + "\n")
            logger.info("input_keyword : " + str(input_keyword) + "\n")
            logger.info("input_evidence : " + str(input_evidence) + "\n")
            raise ValueError("Input Error")

        # 질문 받은 문서와 질문을 기반으로 Dictionary 생성
        input_dict = {"context": input_context, "question": input_question,
                      "keyword": input_keyword, "evidence":input_evidence, "id": ""}

        dataset, examples, features = load_examples(args, tokenizer, evaluate=True, output_examples=True, input_dict=input_dict)

        # 예측값 저장을 위한 리스트 선언
        all_results = []

        predict_dataloader = DataLoader(dataset, batch_size=1)
        logger.info("   Inference Answer.....")
        for batch in progress_bar(predict_dataloader):
            # 평가 모드로 변경
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    # "token_type_ids": batch[2],   # roberta
                    "keyword_labels":batch[6],
                    "evidence_labels":batch[7],
                }

                example_indices = batch[3]     # feature ids
                outputs = model(**inputs)

            for i, example_index in enumerate(example_indices):
                # feature 고유 id로 접근하여 원본 q_id 저장
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                # outputs = [start_logits, end_logits]
                output = [to_list(output[i]) for output in outputs]

                # start_logits: [batch_size, max_length]
                # end_logits: [batch_size, max_length]
                start_logits, end_logits = output

                # q_id에 대한 예측 정답 시작/끝 위치 확률 저장
                result = SquadResult(unique_id, start_logits, end_logits)

                # feature에 종속되는 최종 출력 값을 리스트에 저장
                all_results.append(result)

        # q_id에 대한 N개의 출력 값의 확률로 부터 가장 확률이 높은 최종 예측 값 저장
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.do_lower_case,
            None,  # output_prediction_file,
            None,  # output_nbest_file,
            None,  # output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )
        predict_answer = normalize_answer(predictions[""])

        # 예측 정답 출력
        print("Predict Answer : {}".format(predict_answer))

        # except:
        #     print("Inference Error!")




import json
import torch
import numpy as np
from konlpy.tag import Mecab
import preprocess.datautils.utils as utils

def process_question_multichoice(q_dic:dict):
    q = q_dic['question']
    answer_cand = np.asarray(q_dic['answers'])

    with open('./data/video-narr/video-narr_vocab.json','r') as f:
        vocab = json.load(f)

    return multichoice_encoding_data(vocab, q, 0, answer_cand)

def multichoice_encoding_data(vocab, question,correct_idx, answer_candidate):
    #Encode all questions
    print('Encoding data')
    m = Mecab().morphs
    
    questions_encoded = []
    questions_len =[]
    all_answer_candidate_encoded = []
    all_answer_candidate_lens = []

    question_tokens = m(question)
    question_encoded = utils.encode(question_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
    questions_encoded.append(question_encoded)
    questions_len.append(len(question_encoded))

    answer = 0
    correct_answer = answer

    candidates_encoded = []
    candidates_len = []

    for answer in answer_candidate:
        answer_tokens = m(answer)
        candidate_encoded = utils.encode(answer_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
        candidates_encoded.append(candidate_encoded)
        candidates_len.append(len(candidate_encoded))
    all_answer_candidate_encoded.append(candidates_encoded)
    all_answer_candidate_lens.append(candidates_len)
    
    # Pad encoded questions
    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab['question_answer_token_to_idx']['<NULL>'])

    questions_encoded =  np.asarray(questions_encoded, dtype=np.int32)
    questions_len = np.asarray(questions_len, dtype=np.int32)
    print(questions_encoded.shape)

    # Pad encoded answer candidates
    max_answer_candidate_length = max(max(len(x) for x in candidate) for candidate in all_answer_candidate_encoded)
    for ans_cands in all_answer_candidate_encoded:
        for ans in ans_cands:
            while len(ans) < max_answer_candidate_length:
                ans.append(vocab['question_answer_token_to_idx']['<NULL>'])

    all_answer_candidate_encoded = np.asarray(all_answer_candidate_encoded, dtype=np.int32)
    all_answer_candidate_lens = np.asarray(all_answer_candidate_lens, dtype=np.int32)
    print(all_answer_candidate_encoded.shape)

    glove_matrix = None

    return {
        'question': questions_encoded,
        'question_len': questions_len,
        'ans_candidate': all_answer_candidate_encoded,
        'ans_candidate_len': all_answer_candidate_lens,
        'answer': correct_answer,
        'glove': glove_matrix,
    }
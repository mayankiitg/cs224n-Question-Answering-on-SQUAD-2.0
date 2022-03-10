import numpy as np

data = np.load('data/train.npz')


np.savez('data/smaller_train.npz', context_idxs=data['context_idxs'][:3], 
                                    context_char_idxs=data['context_char_idxs'][:3], 
                                    ques_idxs=data['ques_idxs'][:3], 
                                    ques_char_idxs=data['ques_char_idxs'][:3], 
                                    y1s=data['y1s'][:3], 
                                    y2s=data['y2s'][:3], 
                                    ids=data['ids'][:3])
                                 
                                 
# Then run 
# python train.py -n today --num_workers 0 --train_record_file 'data/smaller_train.npz' --num_epochs 90 --eval_steps 90 --dev_record_file 'data/smaller_train.npz' --dev_eval_file 'data/train_eval.json'


import os
import json
import matplotlib.pyplot as plt
import argparse


def result_extractor(files, results_path, num_samples_per_file):

    sim_avg, sim_avg_hit, total_change, total_word, hit, hit_sim, num_samples_total, num_samples_hit_sim = 0, 0, 0, 0, 0, 0, 0, 0
    for f in files:
        f_path = os.path.join(results_path, f)
        with open(f_path , 'r') as f_t:
            data = json.load(f_t)

        for idx, res in enumerate(data[:-1]):
            num_samples_total += 1
            if res['change']/res['num_word'] < 0.1: # compute metrics for change rate less than 0.1
                if res['success']==4:
                    sim_avg = res['sim'] + sim_avg
                    hit +=1
                    total_change += res['change']
                    total_word += res['num_word']
                    if res['sim'] > 0.8:
                        hit_sim += 1
                        sim_avg_hit = res['sim'] + sim_avg_hit
                        num_samples_hit_sim += 1
            if idx == num_samples_per_file:
                break          
    
    sim = sim_avg / num_samples_total
    # sim_08 = sim_avg_hit / num_samples_hit_sim
    acc = (hit/num_samples_total) * 100.0
    acc_sim_08 = (hit_sim/num_samples_total) * 100.0
    change = (total_change / total_word) * 100.0

    return acc, acc_sim_08, sim, change




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str)
    args = parser.parse_args()
    
    results_path_normal = args.result_path

    num_samples_per_file = 1000

    files_normal = os.listdir(results_path_normal)
    
    pos_normal_files = [f for f in files_normal if 'pos' in f]

    acc, acc_sim_08, sim, change = result_extractor(pos_normal_files, results_path_normal, num_samples_per_file)
    
    print(F'Average success rate: {acc}')
    print(F'Average success rate (when sim>0.8): {acc_sim_08}')
    print(F'Average similarity: {sim}')
    # print(F'Average similarity (when sim>0.8): {sim_08}')
    print(F'Average change rate: {change}')

if __name__=='__main__':
    main()

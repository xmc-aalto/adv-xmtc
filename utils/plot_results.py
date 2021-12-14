
import os
import json
import matplotlib.pyplot as plt
import argparse


def result_extractor(files, results_path):
    acc = {}
    change = {}
    sim = {}
    acc_sim = {}
    change_hit_dict = {}
    for f in files:
        try:
            freq = int(f.split('_')[2])
        except ValueError:
            print('value error')
            continue
        f_path = os.path.join(results_path, f)
        with open(f_path , 'r') as f_t:
            data = json.load(f_t)
        sim_avg = 0
        hit = 0
        hit_sim = 0
        change_hit = 0
        num_word_hit = 0
        for res in data[:-1]:
            if res['change']/res['num_word'] <= 0.1: # compute metrics for change rate less than 0.1
                if res['success']==4:
                    change_hit += res['change']
                    num_word_hit += res['num_word']
                    sim_avg = res['sim'] + sim_avg
                    hit +=1
                    if res['sim'] > 0.8:
                        hit_sim += 1
        
        sim[freq] = sim_avg/hit
        acc[freq] = (hit/len(data[:-1])) * 100.0
        acc_sim[freq] = (hit_sim/len(data[:-1])) * 100.0

        change_hit_dict[freq] = (change_hit/num_word_hit) * 100.0
    
    acc = dict(sorted(acc.items()))
    # change = dict(sorted(change.items()))
    change = 1
    acc_sim = dict(sorted(acc_sim.items()))
    sim = dict(sorted(sim.items()))

    change_hit_dict = dict(sorted(change_hit_dict.items()))

    return acc, change, change_hit_dict, acc_sim, sim




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str)
    args = parser.parse_args()
    
    results_path_normal = args.result_path

    files_normal = os.listdir(results_path_normal)

    pos_normal_files = [f for f in files_normal if 'pos' in f]

    pos_normal_acc, pos_normal_change, pos_normal_change_hit, pos_normal_acc_sim, pos_normal_sim = result_extractor(pos_normal_files, results_path_normal)
    


    plt.plot(list(pos_normal_acc_sim.keys()), list(pos_normal_acc_sim.values()), linewidth=2.0)
    plt.xscale('log')
    plt.xlabel('Frequency')
    plt.ylabel('Success rate')
    plt.show()

if __name__=='__main__':
    main()

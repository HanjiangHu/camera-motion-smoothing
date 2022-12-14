# Code modified based on  https://github.com/AI-secure/semantic-randomized-smoothing
import os
import argparse
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import norm

parser = argparse.ArgumentParser(description='Analyze the real performance from logs')
parser.add_argument("logfile", help="path of the certify.py output")
parser.add_argument("outfile", help="the output path of the report")
parser.add_argument("--budget", type=float, default=0.0,
                    help="for semantic certification, the pre-allocated space for semantic transformations")
parser.add_argument("--step", type=float, default=0.25, help="step size for l2 robustness")
args = parser.parse_args()

def change_alpha(data):
    new_data = data.copy()
    for i in range(len(data)):
        if int(data["correct"][i]):
            radius = data["radius"][i]
            NA_list = []
            for j in range(1001):
                try_radius = 0.05 * norm.ppf(proportion_confint(j, 1000, alpha=2 * 0.01, method="beta")[0])
                if abs(radius - float("{:.3}".format(try_radius))) < 0.000001:
                    NA_list.append(j)
                    print(i, j, radius, float("{:.3}".format(try_radius)))
            # assert len(NA_list) == 1, f"{i}{NA_list}"
            new_radius = 0.05 * norm.ppf(proportion_confint(NA_list[-1], 1000, alpha=2 * 0.001, method="beta")[0])
            new_data["radius"][i] = float("{:.3}".format(new_radius))
    return new_data


if __name__ == '__main__':
    df = pd.read_csv(args.logfile, delimiter="\t")
    print(f'Total: {len(df)} records')
    steps = list()
    nums = list()
    now_step = args.budget
    while True:
        cnt = (df["correct"] & (df["radius"] >= now_step)).sum()
        mean = (df["correct"] & (df["radius"] >= now_step)).mean()
        steps.append(now_step)
        nums.append(mean)
        now_step += args.step
        if cnt == 0:
            break
    steps = [str(s) for s in steps]
    nums = [str(s) for s in nums]
    output = "\t".join(steps) + "\n" + "\t".join(nums)
    print(output)
    print(f'Output to {args.outfile}')
    if not os.path.exists(args.outfile):
        os.makedirs(args.outfile)
    f = open(args.outfile + "/certification_results.txt", 'w')
    print(output, file=f)
    print(f'Clean acc: {df["correct"].sum()}/{len(df)} = {df["correct"].sum()/len(df)}')
    f.close()

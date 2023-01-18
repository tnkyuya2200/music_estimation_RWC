#coding:utf-8
import numpy as np
import sys, os
import json
from glob import glob

def main():
    
    filename_list = glob(sys.argv[1] + "/*.json")

    result_dict = {
        "all": {"vocal": 0, "chords": 0, "average": 0, "deno": 0},
        "noise": {"vocal": 0, "chords": 0, "average": 0, "deno": 0},
        "pitch": {"vocal": 0, "chords": 0, "average": 0, "deno": 0},
        "raw": {"vocal": 0, "chords": 0, "average": 0, "deno": 0},
        "snipped": {"vocal": 0, "chords": 0, "average": 0, "deno": 0},
        "speed": {"vocal": 0, "chords": 0, "average": 0, "deno": 0}
    }
    attr_list = ["vocal", "chords", "average"]
    sum_time = 0
    for filename in filename_list:
        file = open(filename, "r", encoding="utf-8")
        input_dict = json.load(file)
        ID_type = os.path.splitext(os.path.basename(input_dict["test_file"]))[0]
        test_ID = int(ID_type.split("_")[0])
        test_type = ID_type.split("_")[1]
        db_list = input_dict["db"]

        result_dict["all"]["deno"] += 1
        result_dict[test_type]["deno"] += 1

        for attr in attr_list:
            result_dict["all"][attr] += (max(db_list, key=lambda x: x["sim"][attr])["ID"] == test_ID)
            result_dict[test_type][attr] += (max(db_list, key=lambda x: x["sim"][attr])["ID"] == test_ID)
        sum_time += input_dict["elap_time"]

    print("type\tvocal\t\t\tchords\t\t\taverage")
    for test_type, data in result_dict.items():
        score = {}
        deno = data["deno"]
        print(test_type, end="\t")
        for attr in attr_list:
            if deno == 0:
                score[attr] = 0
            else:
                score[attr] = data[attr] / deno * 100
            #print("{: >6.2f}".format(score[attr]*100), "%", "(", data[attr], "/", data["deno"], ")", end="\t")
            print(f"{score[attr]: >8.2f} %  ({data[attr]: >4}/{deno: >4})\t", end="")
        print()
    print(f"\nelapsed time: sum- {sum_time: >.2f} s; ave- {sum_time/len(filename_list): >.2f} s")
        
if __name__ == "__main__":
    main()

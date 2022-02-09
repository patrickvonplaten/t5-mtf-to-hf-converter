#!/usr/bin/env python3

replacements = {
    "lg": "large",
    "sm_": "small_",
    "bs": "base",
    "f": "ff",
    "shared": "sh",
    "_h": "_nh",
    "11B": "xxl",
    "3B": "xl",
}


with open("./all_links_processed.txt", "r") as r, open("./hf_names.txt", "w") as w, open("./hf_names_dict.txt", "w") as g:
    lines = r.readlines()
    for line in lines:
        line = line.strip()
        new_line = "_".join(line.split("_")[2:-2])
        if new_line == "":
            new_line = "base"

        for key, value in replacements.items():
            if key in new_line:
                new_line = new_line.replace(key, value)

        if "_l" in new_line:
            new_line = new_line.replace("_l", "_nl")

        if new_line[0] == "d" and "dl" not in new_line:
            new_line = new_line.replace("d", "dm")

        if new_line[0] == "l" and "large" not in new_line:
            new_line = new_line.replace("l", "nl")

        if new_line[0] == "h":
            new_line = new_line.replace("h", "nh")

        if "_d" in new_line and "dm" not in new_line and "dl" not in new_line:
            new_line = new_line.replace("d", "dm")

        if new_line[-1] == "k":
            new_line = new_line.replace("k", "000")

        if "dl" in new_line and "el" in new_line:
            new_line = "_dl".join(new_line.split("dl"))

        if len(new_line.split("_")) == 1 and new_line not in ["tiny", "mini", "small", "base", "large", "xl", "xxl"]:
            new_line = "base_" + new_line

        new_line = "t5-efficient-" + "-".join(new_line.split("_"))

        w.write(new_line + "\n")
        g.write(line[:-1] + " " + new_line + "\n")

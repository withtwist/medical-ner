def get(output_pos_list, category_names, wordlist):
    outputs = []
    for o in output_pos_list[1:]:
        if o != "":
            o_list = o.split(",")
            o_list = [int(n) for n in o_list]
            outputs.append(o_list)

    start_tag = ["<{}>".format(category_name) for category_name in category_names]
    end_tag = ["</{}>".format(category_name) for category_name in category_names]
    for a in outputs:
        wordlist[a[1]] = start_tag[a[0]] + wordlist[a[1]]
        wordlist[a[2]] = wordlist[a[2]] + end_tag[a[0]]
    return "".join(wordlist)

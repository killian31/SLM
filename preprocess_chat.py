from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def preprocessing_pipeline(
    infile,
    name_dic,
    save=True,
    start_date="11 juillet 2021",
    end_date="22 octobre 2022",
):
    with open(infile, "r") as f:
        text = f.readlines()
    for i in range(len(text)):
        text[i] = text[i].strip()
        text[i] = text[i].strip("\u200e")

    pretext = [elt for elt in text if len(elt) > 2]
    pretext2 = [elt for elt in pretext if "\u200e" not in "".join(elt)]

    # create the dataset

    tab = []

    for elt in pretext2:
        separ = elt.split(" ")
        assert separ[0] != ""
        if separ[0][0] == "[":
            date = separ[0] + " " + separ[1]
            name = separ[2].split(":")[0]
            message = " ".join(separ[3:])
            tab.append([date, name, message])
        else:
            tab[-1][2] += " " + elt

    dt = np.array(tab).reshape(len(tab), 3)
    data = pd.DataFrame(dt, columns=["Date", "Name", "Message"])
    t = []
    for dat in data.Date:
        t.append(dat[1:-1])
    data["Date"] = t
    """
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
    data.set_index("Date", inplace=True)
    """
    return data

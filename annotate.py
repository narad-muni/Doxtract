import random,json
seperator = [": "," "," : ",":- ","\n"]

# annotations = {
#     "classes": ["KEY","VALUE"],
#     "annotations":[
#         ["order number 152542",{"entities":[[0,12,"KEY"],[12,17,"VALUE"]]}],
#     ]
# }

annotations = {
    "classes": ["KEY","VALUE"],
    "annotations":[
        
    ]
}

f = open('data.txt',"r")
out = open('annotated.json',"w+")

for line in f:
    if("----" in line):
        an = []
        sep = seperator[random.randint(0,4)]
        val = line.split("@@@@")
        ent = {"entities":[]}
        line = line.replace("@@@@",sep)
        k = [0,len(val[0]),"KEY"]
        v = [len(val[0])+len(sep),line.index("----"),"VALUE"]
        ent["entities"] = [k,v]
        an.append(line.replace("----","\n").replace("(NewLine)","\n"))
        an.append(ent)
        annotations["annotations"].append(an)
    else:
        an = []
        sep = seperator[random.randint(0,4)]
        val = line.split("@@@@")
        ent = {"entities":[]}
        k = [0,len(val[0]),"KEY"]
        v = [len(val[0])+len(sep),len(val[0])+len(sep)+len(val[1])-1,"VALUE"]
        ent["entities"] = [k,v]
        an.append(line.replace("@@@@",sep).replace("(NewLine)","\n"))
        an.append(ent)
        annotations["annotations"].append(an)

out.write(json.dumps(annotations))
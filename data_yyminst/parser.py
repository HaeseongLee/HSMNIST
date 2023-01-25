import os

path = "/home/roboe/roboe_ws/src/roboeod/script/HSMNIST/data_yyminst/dataset/yymnist_train.txt"

IM_SIZE = 416

with open(path) as f:
    lines = f.readlines()
    for l in lines:
        seg = l.split(" ")
        
        _,name = os.path.split(seg[0])
        id,_ = os.path.splitext(name)
        
        root,_ = os.path.split(os.path.abspath(__file__))        
        save_name = os.path.join(root,"dataset/labels/train",id+".txt")

        with open(save_name,'w') as label:
            for data in seg[1:]:      
                data = data.split(",")                
                cx = (float(data[0])+float(data[2]))/2
                cy = (float(data[1])+float(data[3]))/2
                w = float(data[2]) - float(data[0])
                h = float(data[3]) - float(data[1])
                cx, cy = cx/IM_SIZE, cy/IM_SIZE                
                w, h = w/IM_SIZE, h/IM_SIZE                
                n = int(data[4])

                annotation = ' '.join([str(n), str(cx), str(cy), str(w), str(h)])
                annotation += "\n"
                label.write(annotation)
        # with open(save_name) as label:

        
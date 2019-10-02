import imageio

path="CGAN"
num=30

output_filename="./"+path+".gif"


filenames=[]
for i in range(num):
    filenames.append("./"+path+"/result"+str(i+1)+".png")

frames=[]
for filename in filenames:
    image=imageio.imread(filename)
    frames.append(image)
imageio.mimsave(output_filename,frames,'GIF',duration=0.5)

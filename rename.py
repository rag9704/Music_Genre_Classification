import os
os.chdir(r'C:\Users\Rag9704\Pictures\genres\hiphop')
i=1
for file in os.listdir():
    src=file
    dst="hiphop"+str(i)+".wav"
    os.rename(src,dst)
    i+=1

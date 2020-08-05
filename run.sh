archs="arcface arcface_extend cosface sphereface normface"
for i in $archs
do 
    python train_new.py --draw --arch $i
done;

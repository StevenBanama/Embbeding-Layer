archs="arcface cosface cosface_extend sphereface sphereface_extend normface"
for i in $archs
  do 
    python train_new.py --draw --arch $i
  done;

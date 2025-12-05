python gen_ns_init.py --init disk  --A1 1.0  --R1 1.0  --L 24  --N 768   --T 20.0  --dt 0.005   --save_every 10   --out artifacts   

(This is for real data.)


Train on 2D N-S (for vorticity) under FCNet

python train_window_nets.py --model fcnet --tmin 0.0 --tmax 10.0 --C 4.0 --epochs 20 --iters 100 --M 8192 --microbatch 2048 --lr 2e-3 --amp --device cuda --artifacts artifacts

python compare_surface3d_physical.py --model fcnet --times 4,6,8 --C 4.0 --Nxy 201 --device cuda --artifacts artifacts --out compare_surface_deeponet_t4_6_8.png

python compare_surface3d_physical.py --model fcnet --times 12,14,16,18 --C 4.0 --Nxy 201 --device cuda --artifacts artifacts --out compare_surface_deeponet_t12_14_16_18.png




Train on 2D N-S (for vorticity) under ConcatMLP

python train_window_nets.py --model concat --tmin 0.0 --tmax 10.0 --C 4.0 --tmin 0.0 --tmax 10.0 --epochs 20 --iters 100 --M 8192 --microbatch 2048 --lr 2e-3 --amp --device cuda --artifacts artifacts

python compare_surface3d_physical.py --model concat --times 12,14,16,18 --C 4.0 --Nxy 201 --device cuda --artifacts artifacts --out compare_surface_concat_t12_14_16_18.png
	



Train on 1D Viscous Burgers under FCNet

python train_fcnet_burgers_1d.py --model fcnet --truth artifacts/truth_burgers_1d.npz --t_train_max 10 --nsamples 60000 --epochs 400 --batch 1024 --lr 1e-3 --width 128 --depth 4 --latent 128 --phys_ckpt artifacts/ckpt_fcnet_physical_burgers.pt --ssv_ckpt artifacts/ckpt_fcnet_ssv_burgers.pt







Train on 1D Viscous Burgers under ConcatMLP

python train_fcnet_burgers_1d.py --model concat --truth artifacts/truth_burgers_1d.npz --t_train_max 10 --nsamples 60000 --epochs 400 --batch 1024 --lr 1e-3 --width 128 --depth 4 --latent 128 --phys_ckpt artifacts/ckpt_concat_physical_burgers.pt --ssv_ckpt artifacts/ckpt_concat_ssv_burgers.pt


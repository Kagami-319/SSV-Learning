Sample commands:

Real solution of NS: python gen_ns_init.py --init disk  --A1 1.0  --R1 1.0  --L 24  --N 768   --T 20.0  --dt 0.005   --save_every 10   --out artifacts

Sln of Burgers: python gen_burgers_init.py --xmax 20 --nx 4096 --tfinal 20 --times auto --auto_dt 0.1 --method imex --dt 5e-4 --bc neumann --save truth_burgers_1d.npz


NS Training: python train_window_nets.py --tmin 0.0 --tmax 0.3 --C 4.0 --epochs 10 --iters 80 --M 8192 --microbatch 2048 --lr 2e-3 --amp --device cuda --artifacts artifacts  

NS Plotting with quantitative MSE comparison: python compare_surface3d_physical_with_mse.py --model fcnet --times 0.5,1,1.5  --mse_interval 0,5 --mse_interval_n 100 --C 4 --Nxy 201 --device cuda --artifacts artifacts

Burgers Training: python train_fcnet_burgers_1d.py --model fcnet --truth artifacts/truth_burgers_1d.npz --t_train_max 0.5 --nsamples 2000 --epochs 200 --batch 1024 --lr 1e-3 --width 128 --depth 4 --latent 128 --phys_ckpt artifacts/ckpt_fcnet_physical_burgers.pt --ssv_ckpt artifacts/ckpt_fcnet_ssv_burgers.pt

Burgers Plotting: python compare_burgers_nets_1d.py --truth artifacts/truth_burgers_1d.npz --artifacts artifacts --model concat --times 2.5 --x_clip 8.0 --width 128 --depth 4 --latent 128 --out artifacts/compare_burgers_concat_t2.5.png

Burgers MSE Comparison: python compare_burgers_nets_1d_mse_interval_plot_lines.py --truth artifacts/truth_burgers_1d.npz --artifacts artifacts --model fcnet --mse_interval 0.5,5 --mse_interval_n 100 --x_clip 8.0  --mse_out artifacts/mse_fcnet_0p5_5_n100.csv --mse_plot_out artifacts/mse_fcnet_0p5_5_n100.png

## How to print count histograms:
* python count_visualization.py --exp [the_exp_you_want_to_print]
* python print_outliers.py --exp [the_exp_you_want_to_print]  (right now this works both for counts and for forgetting events!) (TODO, implement the same idea for noisy labels)
* python organize_experiment_analysis.py --exp [the_exp_you_want_to_organize]
* python check_masks.py --exp [the_exp_you_want_to_print]    (recall that this experiment should have corrupted labels!)

Now you have inside the "analysis_experiments" folder (which is inside [the_exp_you_want_to_organize]) all the experiments numbered (accordingly to date) and containing all the pdf and log you need!



## How to print class count histograms:
* python new_data_viz.py --exp [the_exp_you_want_to_print] --base [source directory] --sd [save directory] --plot 'class_count' --arch [architectures you want to consider] --datasets [datasets you want to consider] --hue [plot level comparison] --cfl [file level comparison] --epochs [epochs you want to consider]

Examples:
--hue '{"temperature":"all"}'						-->		one bar for each value of the "temperature" keyword
--hue '{"temperature":[1.0,0.1]}'					-->		one bar for "temperature"==1.0, one for "temperature"==0.1
--cfl '{"normalizer":"all","sampler":"all"}'		-->		one figure for each combination of values assumed by "normalizer" and "sampler" keywords (considering all values in the experiments for both keywords)		
--cfl '{"normalizer":"all","sampler":["tunnel"]}'	-->		one figure for each combination of values assumed by "dataset" and "arch" keywords (considering all values in the experiments for "normalizer" and only the value "tunnel" for "sampler")		

Note that if "arch" is not passed as hue argument, it must be included in cfl (same for "dataset") [should be fixed in the near future]. In practice:
case1) --hue '{"arch":"all"}'			-->		--cfl '{"dataset":"all","sampler":"all"}'
case2) --hue '{"dataset":"all"}'		-->		--cfl '{"arch":"all","sampler":"all"}'
case3) --hue '{"temperature":"all"}'	-->		--cfl '{"dataset":"all","arch":"all","sampler":"all"}'

Note that --arch, --datasets and --epochs can have multiple arguments, which must be passed separated with a blank space. In practice:
--arch "resnet18" "resnet34"
--datasets "cifar10" "imagenet_lt"
--epochs 159 160 161

For aquarium1 and exp "CVPR_sampler_resnet18_cifar10", use:
--exp 'CVPR_sampler'
--base '/mnt/DATA/matteo/results/information_sampler/'
--sd '/mnt/DATA/matteo/results/information_sampler/CVPR_sampler_resnet18_cifar10/analysis_experiments/'
--plot 'class_count' 
--arch 'resnet18' 
--datasets 'cifar10' 
--hue '{"temperature":"all"}' 
--cfl '{"dataset":"all","arch":"all","normalizer":"all"}' 
--epochs 0 59 60 61 119 120 121 159 160 161 179

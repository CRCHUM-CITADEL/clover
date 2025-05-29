# FinDiff

## Official repository
https://github.com/sattarov/FinDiff

## License
MIT License

## Changes
* Local imports paths
* Add __init__.py
* Add findiff.py
* Remove unused libraries
* Fix issue when no label is provided in the dataloader for training
* Add last step in time steps sampling
* Include beta_hat for sampling at the first step when adding noise
* Reverse input and target in loss function
* Fix time steps sampling in reverse diffusion process
* Fix alpha, alphas_hat and beta sampling in the reverse diffusion process
* Fix random noise generation at steps 1 in the reverse diffusion process
* Add device parameter to generate_samples function
* Fix issue in generate_samples function when label is None
* Modify train function
* Implement differential privacy
* Create global functions for categorical variables embedding


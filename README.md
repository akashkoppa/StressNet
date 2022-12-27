# StressNet
This repository contains codes and data used for developing the next version of StressNet.
Author - Akash Koppa
Contact - akash.koppa@ugent.be

## Tasklist of improvements v2.0
- [ ] Use a better model for partitioning transpiration.
- [ ] Incorporate plant traits.
- [ ] Incorporate GLEAM-Hydro along with better rooting depth.
- [ ] Develop the explainable AI (XAI) component of StressNet.
- [ ] Bayesian Deep Learning to quantify uncertainty.
- [ ] Generalizable model.

## Partitioning of Transpiration
Explore the partitioning methodology described here: [Link to Reference Paper](https://www.sciencedirect.com/science/article/pii/S0168192321004767#!). This methodology might need more information than available from the FLUXNET database. 

Alternatively, explore the use of the TEA algorithm described here: [Link to Reference Paper](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018JG004727).

## GLEAM-Hydro
Update GLEAM according to Petra's GLEAM-Hydro for groundwater access and Shujie's rooting depth map for improved groundwater access.

## Plant Traits
Plant traits are available from different sources:
1. TRY - global plant trait database [Link to Database](https://www.try-db.org/TryWeb/Home.php).
2. Plant hydraulic traits from Maurizio Mencuccini [Link to Database](https://figshare.com/articles/dataset/Adaptation_and_coordinated_evolution_of_plant_hydraulic_traits_/12625418/1).
3. Upscaled leaf traits from University of Valencia [Link to Database](https://isp.uv.es/code/try.html).
4. Upscaled plant traits from University of Valencia [Still Under Developement?](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiY8fO1-OH1AhWkgP0HHcGqBPoQFnoECAQQAQ&url=https%3A%2F%2Fmeetingorganizer.copernicus.org%2FEGU21%2FEGU21-15835.html%3Fpdf&usg=AOvVaw0ITWUVUjkKrS4z0VNTHinB) 

NOTE: Spatially continuous upscaled plant hydraulic traits are currently not available. Wait till Alvaro Moreno can use Mencuccini's database to upscale these. 
 
## Explainable AI (XAI)
Use the [slundberg/shap](https://github.com/slundberg/shap) repository to apply different XAI methods to derive the importance of covariates

## Bayesian Deep Learning
Either use implicit Bayesian deep learning methods or use Monte Carlo simulations. Too early to decide. 

## Generalizable Model
Currently the model is biased towards regions which have the most number of flux towers. Explore methodologies of developing models which can predict out-of-sample (example: [Link to Reference Paper](https://arxiv.org/abs/2112.08440)


Code for the paper "Guided Weak Supervision for Action Recognition with Scarce Data to Assess Skills of Children with Autism", AAAI 2020.

- tensorflow = 1.14.0
- python = 3.5 or higher
- keras = 2.2.5

## Data Preprocessing
Generate optical flow(using TVL1 algorithm) and RGB frames for the videos using *Preprocess_threads.py*. *utils.py* can be used to create data split.

## Baseline training
Use I3D to train baseline classifier. *train_flow.py* is used to train flow stream and *train_rgb.py* train RGB stream of a two stream network.

## Guided Weak Supervision(GWS)
Use baseline model(their weights) to match modes of the source using *cluster_flow_clips_2s_mode_matched.py* file.
*visualization* folder contain examples depicting concept of matching classes in optical flow space.

## Action localization
If source and target datasets video samples differ in time duration, actions can be localized in a bigger source video using 
*create_2s_clips.py*.

Re-train the baseline with localized-mode matched samples using *train_flow.py*. Only flow stream is re-trained. Some of the
initial layers(a hyperparameter) are freezed during re-training.

## Directional Regularization(DR)
Train a classifier(I3D) only with only mode matched source samples using *freezed_train_flow_kins_1.py*. It re-trains only 
few penultimate layers of I3D pre-trained on Imagenet and Kinetics dataset. Use their weights(fixed) to perforrm DR using *freezed_custom_train_flow_weights_sim_loss_spectral.py*. This is the final trained model.

## Evaluation
Evaluate on test data using weights of final trained model on flow-stream using *evaluate.py*. Use baseline RGB and re-trained flow-stream model
for evaluation.

GWS-DR using TSN will be released soon.

## Results
| Model/Method | Accuracy |
| :---:         |     :---:      |          
| [I3D](https://arxiv.org/pdf/1705.07750.pdf)   | 69.3%    | 
| [TSN](https://wanglimin.github.io/papers/WangXWQLTV_ECCV16.pdf)    |   68.0%     | 
| I3D+GWS    |  74.3%     | 
| TSN+GWS    | 71.6%  | 
| I3D+DR     | 71.3%  | 
| TSN+DR      | 70.1% | 
| I3D+GWS+DR    | 75.1% | 
| TSN+GWS+DR      | 72.5% | 

## Citation

If you find our work useful, please consider citing our paper.

```javascript
@article{pandey2019guided,

  title={Guided weak supervision for action recognition with scarce data to assess skills of children with autism},
  
  author={Pandey, Prashant and AP, Prathosh and Kohli, Manu and Pritchard, Josh},
  
  journal={arXiv preprint arXiv:1911.04140},
  
  year={2019}
  
}
```


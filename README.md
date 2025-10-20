# Honours project on Exercise Posture Analyses Using Computer Vision

## Abstract 


Exercise is essential for improving overall well-being; however, performing exercises with poor posture can limit these benefits and increase the risk of injury.
Hence, maintaining correct posture during exercises is crucial, as it enhances performance and prevents injuries. Typical posture correction methods include personal trainers, self-analyses, and sensor-based approaches; however, these methods can be costly and prone to bias. Therefore, there is a need for automated systems that can detect posture errors and provide feedback on the type of error.

This research explores a pose-based approach for exercise recognition and posture analysis using deep learning techniques on the EC3D and Fitness-AQA datasets, thereby enhancing exercise safety through automated posture feedback. The proposed system utilises the MediaPipe framework to extract keypoints from RGB videos, which are then processed using a Spatial-Temporal Graph Convolutional Network (ST-GCN) to classify the exercise performed and a separate ST-GCN model to analyse the posture.


The proposed system achieved near-perfect results for the exercise recognition task on both datasets. For the posture analysis task, the model achieved 86.84\% accuracy on the EC3D dataset and 79.06\% accuracy in detecting knee errors for overhead press on the Fitness-AQA dataset. Further experiments demonstrated that the proposed model is more effective in analysing single exercise postures than in analysing multi-exercise postures. The proposed model was found to be effective at classifying exercises and analysing posture on the EC3D dataset. However, performance declined on the Fitness-AQA dataset due to the increased complexity introduced by varying camera angles, natural gym environments, and occlusions. Overall, the proposed ST-GCN models for exercise recognition and posture analysis were proven to be effective at exercises and posture classification.

## Datasets
* Fitness-AQA
* EC3D

## References
Parmar, P., Gharat, A., and Rhodin, H. Domain knowledge-informed self-supervised representations for workout form assessment. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXVIII, pages 105–123. Springer, 2022.

Zhao, Z., Kiciroglu, S., Vinzant, H., Cheng, Y., Katircioglu, I., Salzmann, M., and Fua, P. 3d pose based feedback for physical exercises. In Proceedings of the Asian Conference on Computer Vision, pages 1316–1332. 2022

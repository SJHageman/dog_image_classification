# dog_image_classification

# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This was done on the provided dog breed classication data set.


## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.

### Access
The data was uploaded to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

For this model, the pretrained resnet18 pytorch model was used.  The hyperparameters that were tuned were the batch_size, epochs, and learning rate.

This is a screenshot of the completed hyperparameter training jobs:
![hpo](https://github.com/SJHageman/dog_image_classification/blob/main/hpo_tuning_job.PNG)

and the corresponding best estimator parameters:
![best_est](https://github.com/SJHageman/dog_image_classification/blob/main/hpo_best_estimator.png)

Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker
To perform debugging and profiling, I used Sagemaker Debugger and Profiler to generate a debugging and profiling report.

These were the debugging and profiling rules that were used:
rules = [
    Rule.sagemaker(rule_configs.loss_not_decreasing()),
    ProfilerRule.sagemaker(rule_configs.LowGPUUtilization()),
    ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
    Rule.sagemaker(rule_configs.vanishing_gradient()),
    Rule.sagemaker(rule_configs.overfit()),
    Rule.sagemaker(rule_configs.overtraining()),
    Rule.sagemaker(rule_configs.poor_weight_initialization()),
]

One example plot is the CPU untilization of the instance during training, shown here:
![util](https://github.com/SJHageman/dog_image_classification/blob/main/cpu_utilization.png)

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?
Here is the profiling report: [link](https://github.com/SJHageman/dog_image_classification/blob/main/profiler-report.html)

One of the main issues raised by the profiler report is that my batch_size was likely too small, and this lead to underutilization of the instance that was used for the training job.  So, either the batch size should have been adjusted, or a smaller instance should have been used.

This is snipped from the profiler report showing that:
![profiler](https://github.com/SJHageman/dog_image_classification/blob/main/profiler-report.png)


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
Once the model was trained, it was deployed to an endpoint.  This was done using the PyTorchModel framework, and a inference.py entry_point.  model_fn, input_fn, and predict_fn functions were developed to be used with this particular model.  And the default output_fn was also used.

To query the endpoint, an image needs to be opened as a numpy array, and then passed to the predictor.predcit function.  The response from the endpoint gives the probability of the dog image being in one of the 133 dog breeds within the dataset.

An example prediction is shown here:
![pred](https://github.com/SJHageman/dog_image_classification/blob/main/prediction_from_endpoint.png)

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.
This is the depoloyed endpoint:
![end](https://github.com/SJHageman/dog_image_classification/blob/main/endpoint.png)


## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.

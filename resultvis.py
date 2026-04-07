from braket.aws import AwsQuantumTask

task = AwsQuantumTask("arn:aws:braket:us-west-1:043439726633:quantum-task/1dd973a7-b17e-45ab-8d85-f1e401d73c18")

result = task.result()

print(result.measurement_counts)
print(result.measurement_probabilities)
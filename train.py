import cntk as C


@C.Function
def create_criterion(z, targets):
    loss = C.losses.cross_entropy_with_softmax(z, targets)
    metric = C.metrics.classification_error(z, targets)
    
    return loss, metric

def create_datasource(filename, sweeps=C.io.INFINITELY_REPEAT):
    image_transforms = [
        C.io.transforms.scale(224, 224, 3),
    ]

    streams = C.io.StreamDefs(
        image=C.io.StreamDef('image', transforms=image_transforms),
        label=C.io.StreamDef('label', shape=2)
    )

    serializer = C.io.ImageDeserializer(filename, streams)

    return C.io.MinibatchSource(serializer, max_sweeps=sweeps)


model_file = 'ResNet18_ImageNet_CNTK.model'
base_model = C.load_model(model_file)

features_node = C.logging.graph.find_by_name(base_model, 'features')
last_node = C.logging.graph.find_by_name(base_model, 'z.x')

cloned_layers = C.combine([last_node.owner]).clone(
        C.CloneMethod.freeze, 
        { features_node: C.placeholder(name='features') })

features_input = C.input_variable((3,224,224), name='features')
normalized_features = features_input - C.Constant(114)

z = cloned_layers(normalized_features)

output_layer = C.layers.Dense(2, activation=C.ops.softmax, name='output')

z = output_layer(z)

targets_input = C.input_variable(2)
criterion = create_criterion(z, targets_input)
learner = C.learners.sgd(z.parameters, lr=0.01)

data_source = create_datasource('data/train/mapping.txt')

progress_writer = C.logging.ProgressPrinter(0)

input_map = {
    features_input: data_source.streams.image,
    targets_input: data_source.streams.label 
}

criterion.train(data_source,
           max_epochs=1,
           minibatch_size=64,
           epoch_size=60000,
           parameter_learners=[learner],
           model_inputs_to_streams=input_map,
           callbacks=[progress_writer])

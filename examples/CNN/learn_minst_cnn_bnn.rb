require_relative '../../lib/fast_neurons'
require_relative '../../lib/mnist_loader'
require 'gnuplot'


include FastNeuronsCNN

puts "Loading images"

# Load MNIST.
 mnist = MNISTLoader.new("../../assets/t10k-images-idx3-ubyte.gz", "../../assets/t10k-labels-idx1-ubyte.gz")
images = mnist.load_images
labels = mnist.load_labels

# Sets the width and height of the input.
width = height = 28

# Normalize input values.
inputs = images.map do |image|
    mnist.binarize(mnist.normalize(image,-1.0..1.0),-1.0,1.0).flatten
end
inputs = inputs.map do |input|
    Numo::DFloat.asarray(input).reshape!(width,height)
end
# puts "inputs[0] = #{inputs[0].inspect}"
# Normalize the expected results
expects = labels.map { |label| res = [-1.0] * 10 ; res[label] = 1.0 ; res }
expects = expects.map { |expect| Numo::DFloat.asarray(expect) }
# puts "expects[0] = #{expects[0].inspect}"

# Generate the samples.
samples = inputs.zip(expects)
samples.shuffle
# train = samples[0..samples.size*2/3]
train = samples[0..99]
# test  = samples[samples.size*2/3+1..-1]
test  = samples[0..99]


# Check the creation of CNN.
# cnn = CNN.new([width,height],
#               [ Dense[256, bnn: true], Activation[Sign],
#                 Dense[10, bnn: true], Activation[Sign] ], rate: 0.01)
# cnn = CNN.new([width,height], [
#                 Convolution[size: [3,3], num: 1, bnn: true], 
#                 Activation[Sign],
#                 Dense[10, bnn: true], Activation[Sign] ], rate: 0.01)
cnn = CNN.new([width,height], [
                Convolution[size: [3,3], num: 4, bnn: true], 
                Pooling[size: [2,2], func: :max], 
                Activation[Sign],
                Convolution[size: [3,3], num: 1, bnn: true],
                # Pooling[size: [2,2], func: :max], 
                Activation[Sign],
                Dense[10, bnn: true], Activation[Sign] ], rate: 0.01)


puts "Structure:"
puts cnn.structure
puts

best_success_rate = 0.0
best_state = {}

puts "Training..."
2000.times do |epoch|

    success = 0
    failure = 0

    train.each do |sample|
        cnn.update_input(sample[0])
        cnn.update_expect(sample[1])

        cnn.forward

        # puts "Output: #{cnn.output.inspect}"
        # puts "Expect: #{sample[1].inspect}"

        cnn.update_loss
        cnn.update_grad

        # puts "Loss: #{cnn.loss.inspect}"

        # puts "Epoch #{epoch}, Loss: #{cnn.loss.inspect}"
        # puts "Grad: #{cnn.grad.inspect}"

        cnn.backward
        # puts "Delta: #{cnn.delta.inspect}"

        cnn.reset_loss
        cnn.reset_grad


        if cnn.output.argmax == sample[1].argmax then
            success += 1
        else
            failure += 1
        end

        # print("=")

    end
    success_rate = (success*100.0)/(success+failure)
    puts "Epoch #{epoch}, success rate: #{success_rate}"

    if (success_rate > best_success_rate) then
        # Best sucess rate then save current state.
        best_state = cnn.state
        best_success_rate = success_rate
    end
    if success_rate == 100.0 then
        # Cannot be better, stop here.
        break
    end
end

# Serialize the best state.
File.open("cnn_state.json","w") { |f| f.write(best_state.to_json) }


success = 0
failure = 0

# Set the cnn to the best state.
cnn.state = best_state

puts "\nTesting..."

test.each do |sample|
    cnn.update_input(sample[0])

    cnn.forward

    # puts "Output: #{cnn.output.inspect}"
    # puts "Expect: #{expects[idx].inspect}"

    cnn.update_loss

    # puts "Loss: #{cnn.loss.inspect}"

    cnn.reset_loss

    if cnn.output.argmax == sample[1].argmax then
        success += 1
    else
        failure += 1
    end
    puts "Sucess rate: #{(success * 100.0) / (success+failure)}%"
end



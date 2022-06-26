require_relative '../../lib/fast_neurons'
require_relative '../../lib/mnist_loader'
require 'gnuplot'


=begin

m = Numo::DFloat.new(1024,1024).seq*2
f = Numo::DFloat.new(3,3).seq
p = Numo::DFloat.zeros(2,2)
# d = Numo::DFloat.zeros(1022,1022)

puts "m:"
puts m.inspect
puts "f:"
puts f.inspect
puts "p:"
puts p.inspect

puts Time.now

conv = FastNeurons::Convolve.new(m.shape,f.shape)
puts "Convoler built."

d = nil

puts Time.now
100.times { d = conv.(m,f) }
puts Time.now

puts "d:"
puts d.inspect

puts Time.now
pool = FastNeurons::Pool.new(m.shape,p.shape,:min)
puts "Pooler built."

d2 = nil
puts Time.now
100.times { d2 = pool.(m) }
puts Time.now

puts "d2:"
puts d2.inspect

=end

include FastNeuronsCNN

puts "Loading images"

# Load MNIST.
 mnist = MNISTLoader.new("../../assets/t10k-images-idx3-ubyte.gz", "../../assets/t10k-labels-idx1-ubyte.gz")
images = mnist.load_images
labels = mnist.load_labels

# Sets the width and height of the input.
width = height = 28

# Normalize input values.
inputs = images.map { |image| mnist.normalize(image).flatten }
inputs = inputs.map { |input| Numo::DFloat.asarray(input).reshape!(width,height) }

# Normalize the expected results
expects = labels.map { |label| res = [0.0] * 10 ; res[label] = 1.0 ; res }
expects = expects.map { |expect| Numo::DFloat.asarray(expect) }

# Generate the samples.
samples = inputs.zip(expects)
samples.shuffle
train = samples[0..samples.size*2/3]
# train = samples[0..99]
test  = samples[samples.size*2/3+1..-1]
# test  = samples[0..99]


# Check the creation of CNN.
# cnn = CNN.new([width,height],
#               [ Dense[30], Activation[Sigmoid],
#                 Dense[10], Activation[Sigmoid] ], rate: 0.01)
# cnn = CNN.new([width,height], [
#                 Convolution[size: [3,3], num: 1], Activation[LeakyReLU],
#                 Dense[10], Activation[Sigmoid] ], rate: 0.1)
cnn = CNN.new([width,height], [
                Convolution[size: [3,3], num: 4], 
                Pooling[size: [2,2], func: :max], 
                Activation[LeakyReLU],
                Convolution[size: [3,3], num: 1],
                # Pooling[size: [2,2], func: :max], 
                Activation[LeakyReLU],
                Dense[10], Activation[Sigmoid] ], rate: 0.01)


puts "Structure:"
puts cnn.structure
puts


puts "Training..."
2000.times do |epoch|

    success = 0
    failure = 0

    train.each do |sample|
        # idx = Random.rand(inputs.size)
        # puts "\n\n==============================\n"
        # puts "One-step learn for sample ##{idx}..."

        # cnn.update_input(inputs[idx])
        # cnn.update_expect(expects[idx])
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
    puts "Epoch #{epoch}, success rate: #{(success*100.0)/(success+failure)}"
end

success = 0
failure = 0

puts "\nTesting..."

test.each do |sample|
    # idx = Random.rand(inputs.size)
    # puts "\n\n==============================\n"
    # puts "One-step learn for sample ##{idx}..."

    # cnn.update_input(inputs[idx])
    # cnn.update_expect(expects[idx])
    cnn.update_input(sample[0])
    # cnn.update_expect(sample[1])

    cnn.forward

    # puts "Output: #{cnn.output.inspect}"
    # puts "Expect: #{expects[idx].inspect}"

    cnn.update_loss
    # cnn.update_grad

    # puts "Loss: #{cnn.loss.inspect}"
    # puts "Grad: #{cnn.grad.inspect}"

    # cnn.backward
    # puts "Delta: #{cnn.delta.inspect}"

    cnn.reset_loss
    # cnn.reset_grad

    if cnn.output.argmax == sample[1].argmax then
        success += 1
    else
        failure += 1
    end
    puts "Sucess rate: #{(success * 100.0) / (success+failure)}%"
end



require_relative '../../lib/fast_neurons'
require_relative '../../lib/mnist_loader'
require 'gnuplot'

puts "Loading images"

# Load MNIST.
mnist = MNISTLoader.new("../../assets/t10k-images-idx3-ubyte.gz", "../../assets/t10k-labels-idx1-ubyte.gz")
images = mnist.load_images
labels = mnist.load_labels

puts "Initializing network"
loss = []

# Initialize a neural network.
# nn = FastNeurons::BNN.new([784, 15, 784], [:Sigmoid, :Sigmoid], :SquaredError)
nn = FastNeurons::BNN.new([784, 200, 10], [:Sign, :Sign], :SquaredError)

# Load learned network.
nn.load_network("network_bnn.json")

# Binarize the network.
nn.binarize!
# puts "nn.biases=#{nn.biases}"
# puts "nn.weights=#{nn.weights}"

# Normalize pixel values.
# imgs = images.map { |image| mnist.normalize(image,-1.0..1.0).flatten }
imgs = images.map do |image|
    mnist.binarize(mnist.normalize(image,-1.0..1.0),-1.0,1.0).flatten
end

# Normalize the labels
lbs = labels.map { |label| res = [-1.0] * 10 ; res[label] = 1.0 ; res }

puts "Runnning..."

count = 0
success = 0
failure = 0
imgs.each.with_index do |inputs,index|

    nn.input(inputs,lbs[index]) # Input training data and teaching data.
    nn.propagate # Compute feed forward propagation and backpropagation.

    mnist.print_ascii(inputs,-1.0..1.0) # Output training data.
    puts "exp   = #{lbs[index]}"
    puts "guess = #{nn.get_outputs.to_a.flatten}"
    count += 1
    if (lbs[index] == nn.get_outputs.to_a.flatten) then
        success += 1 
        puts "Success"
    else
        failure += 1
    end
    #nn.compute_loss
    #loss << nn.get_loss
    #nn.initialize_loss
end
puts "Number: #{count}, success: #{success}, failure: #{failure}, rate: #{success.to_f/count*100}%"


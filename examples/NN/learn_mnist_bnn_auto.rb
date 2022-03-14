require_relative '../../lib/fast_neurons'
require_relative '../../lib/mnist_loader'
require 'json'
require 'gnuplot'

puts "Loading images"

# Load MNIST.
mnist = MNISTLoader.new("../../assets/t10k-images-idx3-ubyte.gz", "../../assets/t10k-labels-idx1-ubyte.gz")
images = mnist.load_images

puts "Initializing network"
loss = []

# Initialize a neural network.
# nn = FastNeurons::BNN.new([784, 15, 784], [:Sigmoid, :Sigmoid], :SquaredError)
msize = 256
nn = FastNeurons::BNN.new([784, msize, 784], [:Sign, :Sign], :SquaredError)

# Set learning rate.
nn.set_learning_rate(0.001)

# Set mini-batch size.
#nn.set_batch_size(1)

# Set up the parameters to random values.
# nn.randomize(:GlorotNormal, :Zeros)
nn.randomize(:GlorotNormal, :Zeros)

# Load learned network.
#nn.load_network("network_bnn_auto.json")

# Normalize pixel values.
# imgs = images.map { |image| mnist.normalize(image).flatten }
imgs = images.map { |image| mnist.normalize(image,-1.0..1.0).flatten }
middle_data = []

puts "Runnning..."

# learning
# An Autoencoder is shown below as a sample.
20.times do |epoch|
    count = 0
    imgs.each.with_index do |inputs,index|

        nn.input(inputs,inputs) # Input training data and teaching data.
        nn.run(1) # Compute feed forward propagation and backpropagation.
        middle_data << nn.get_outputs(1).to_a

        # mnist.print_ascii(inputs,-1.0..1.0) # Output training data.
        # mnist.print_ascii(nn.get_outputs,-1.0..1.0) # Output the output of neural network.
        count += 1
        if (count % 100 == 0) then
            puts "epoch=#{epoch} count=#{count}"
            mnist.print_ascii(nn.get_outputs,-1.0..1.0) # Output the output of neural network.
        end
        #nn.compute_loss
        #loss << nn.get_loss
        #nn.initialize_loss
    end
    # Save learning state after each epoch.
    nn.save_network_from(1,"network_bnn_auto.json") # save learned network
end

puts "Understood!"
# nn.save_network("network_bnn_auto.json") # save learned network
# gets
#
# confirmation of network
check_data = []
10.times do
    # input_data = msize.times.map{rand()}
    input_data = middle_data.sample
    nn.input_to(1,input_data)
    nn.propagate_from(1)
    # mnist.print_ascii(nn.get_outputs)
    mnist.print_ascii(nn.get_outputs,-1.0..1.0)
    puts
    check_data << { "input_data" => input_data.to_a.flatten,
            "teach_data" => nn.get_outputs.to_a.flatten }
end

# Save the confirmation run.
File.open("check_bnn_auto.json","w+") do |f|
    f.puts(JSON.pretty_generate(check_data))
end

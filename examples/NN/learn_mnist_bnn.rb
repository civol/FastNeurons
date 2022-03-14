require_relative '../../lib/fast_neurons'
require_relative '../../lib/mnist_loader'
require 'json'
require 'gnuplot'

puts "Loading images"

# Load MNIST.
mnist = MNISTLoader.new("../../assets/t10k-images-idx3-ubyte.gz", "../../assets/t10k-labels-idx1-ubyte.gz")
images = mnist.load_images
labels = mnist.load_labels

puts "Initializing network"
loss = []

# epochs = 50
epochs = 100
msize = 448

# Initialize a neural network.
# nn = FastNeurons::BNN.new([784, 15, 784], [:Sigmoid, :Sigmoid], :SquaredError)
# nn = FastNeurons::BNN.new([784, 101, 10], [:Sign, :Sign], :SquaredError)
nn = FastNeurons::BNN.new([784, msize, 10], [:Sign, :Sign], :SquaredError)

# Set learning rate.
nn.set_learning_rate(0.001)

# Set mini-batch size.
#nn.set_batch_size(1)

# Set up the parameters to random values.
# nn.randomize(:GlorotNormal, :Zeros)
nn.randomize(:GlorotNormal, :Zeros)

# Load learned network.
# nn.load_network("network_bnn.json")

# Normalize pixel values.
# imgs = images.map { |image| mnist.normalize(image).flatten }
imgs = images.map do |image|
    mnist.binarize(mnist.normalize(image,-1.0..1.0),-1.0,1.0).flatten
end


# Normalize the labels
lbs = labels.map { |label| res = [-1.0] * 10 ; res[label] = 1.0 ; res }

# Save the data.
File.open("data_bnn.json","w+") do |f|
    data = []
    imgs.zip(lbs) do |img,lb| 
        data << { "input_data" => img, "teach_data" => lb }
    end
    f.puts(JSON.pretty_generate(data))
end


puts "Runnning..."

# learning
# An Autoencoder is shown below as a sample.
old_success = 0
epochs.times do |epoch|
    count = 0
    success = 0
    # puts "epoch=#{epoch}"
    imgs.each.with_index do |inputs,index|

        nn.input(inputs,lbs[index]) # Input training data and teaching data.
        nn.run(1) # Compute feed forward propagation and backpropagation.

        # mnist.print_ascii(inputs,-1.0..1.0) # Output training data.
        count += 1
        if (lbs[index] == nn.get_outputs.to_a.flatten) then
            success += 1
        end
        # puts "got output=#{nn.get_outputs.to_a.flatten}"
        if (count % 100 == 0) then
            puts "epoch=#{epoch} count=#{count} success=#{success} rate=#{success.to_f/count*100}%"
        end
        # puts "exp   = #{lbs[index]}"
        # puts "guess = #{nn.get_outputs.to_a.flatten}"
        # mnist.print_ascii(nn.get_outputs,-1.0..1.0) # Output the output of neural network.
        #nn.compute_loss
        #loss << nn.get_loss
        #nn.initialize_loss
    end
    # Save learning state after each epoch if better success.
    if (success > old_success) then
        puts "Saving network for rate=#{success.to_f/count*100}%"
        nn.save_network("network_bnn.json") # save learned network
        old_success = success
    end
    # nn.save_data("data_bnn.json") # Save the input and training data.
end

puts "Understood!"
# nn.save_network("network_bnn.json") # save learned network
# gets
#
# Restores the best network for confirming.
nn.load_network("network_bnn.json")
check_data = []
# confirmation of network
10.times do
    input_data = imgs.sample
    nn.input(input_data)
    nn.propagate
    # puts "inputs= #{input_data.map{|i| i >=0 ? "1" : "0"}.join}"
    puts "layer[0][0] weights= #{nn.weights[0][0].map{|i| i >=0 ? "1" : "0"}.join}"
    puts "layer[1][0] weights= #{nn.weights[1][0].map{|i| i >=0 ? "1" : "0"}.join}"
    # puts "layer[1] biases= #{nn.biases[1]}"
    puts "layer[0] output: #{nn.get_outputs(1).to_a}"
    # puts "layer[1] z: #{nn.get_z(1).to_a}"
    puts "layer[1] output: #{nn.get_outputs(2).to_a}"
    check_data << { "input_data" => input_data,
            "teach_data" => nn.get_outputs.to_a.flatten }
end

# Save the confirmation run.
File.open("check_bnn.json","w+") do |f|
    f.puts(JSON.pretty_generate(check_data))
end

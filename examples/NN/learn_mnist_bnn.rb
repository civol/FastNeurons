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
nn = FastNeurons::BNN.new([784, 800, 10], [:Sign, :Sign], :SquaredError)

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

puts "Runnning..."

# learning
# An Autoencoder is shown below as a sample.
50.times do |epoch|
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
  # Save learning state after each epoch.
  nn.save_network("network_bnn.json") # save learned network
end

puts "Understood!"
# nn.save_network("network_bnn.json") # save learned network
# gets
# 
# # confirmation of network
# 10.times do
#   nn.input_to(1,15.times.map{rand()})
#   nn.propagate_from(1)
#   # mnist.print_ascii(nn.get_outputs)
#   mnist.print_ascii(nn.get_outputs,-1.0..1.0)
# end
# 
# num = Array.new(loss.size){ |i| i }
# 
# Gnuplot.open do |gp|
#   Gnuplot::Plot.new( gp ) do |plot|    
#     plot.terminal "png"
#     plot.output "learning_curve_mnist.png"
#     plot.xlabel "Steps"
#     plot.ylabel "Loss"
#     plot.yrange "[0:#{loss.max}]"
#     plot.xrange "[0:#{loss.size}]"
# 
#     plot.data << Gnuplot::DataSet.new( [num, loss] ) do |ds|
#       ds.with = "lines"
#       ds.linecolor = "black"
#       ds.linewidth = 3
#       ds.notitle
#     end
#   end
# end

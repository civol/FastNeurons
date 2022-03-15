require_relative '../../lib/fast_neurons'
require_relative '../../lib/clickbait_loader'
require 'gnuplot'

puts "Loading texts..."

# Load the text.
dataset = ClickBaitLoader.new("../../assets/clickbait_data.gz", "../../assets/non_clickbait_data.gz")
dataset.load_clickbaits
dataset.load_non_clickbaits
clickbaits     = dataset.get_clickbait_vectors
non_clickbaits = dataset.get_non_clickbait_vectors

# Create the samples.
samples = (clickbaits.map {|cb|  [cb,  [1.0,0.0]]}) + 
      (non_clickbaits.map {|ncb| [ncb, [0.0,1.0]]})
# Mix them.
samples.shuffle!

epochs = 500
train_size = (samples.size / 1.5).to_i
isize = samples[0][0][0].size
msize = 128

puts "Epoch             = #{epochs}"
puts "Number of samples = #{samples.size}"
puts "Train size        = #{train_size}" 

puts "Initializing network"
loss = []

# Initialize the reservoir.
reservoir = FastNeurons::Reservoir.new(isize,msize, gamma: 0.8, density: 0.2)

# Save the state of the reservoir.
startR = reservoir.R

# Initialize the neural network connected to the reservoir.
# nn = FastNeurons::NN.new([msize,2], [:Sigmoid], :MeanSquare)
nn = FastNeurons::NN.new([msize,15,2], [:Sigmoid, :Sigmoid], :MeanSquare)
# nn = FastNeurons::NN.new([784,15,10], [:Sigmoid,:Sigmoid], :SquaredError)
# nn = FastNeurons::NN.new([784,15,10], [:Tanh,:Tanh], :MeanSquare)

# Set learning rate.
nn.set_learning_rate(0.01)

# Set mini-batch size.
#nn.set_batch_size(1)

# Set up the parameters to random values.
nn.randomize(:GlorotNormal, :Zeros)

# Load learned network.
#nn.load_network("network.json")


puts "Runnning..."

puts "Preparing samples..."
samples.shuffle!
train_samples = samples[0..train_size-1]
test_samples  = samples[train_size..-1]

puts "Precomputing the reservoir..."

# Precompute the reservoir results.
nRs = []
train_samples.each_with_index do |sample,index|
    # Restores the reservoir.
    nR = reservoir.R = startR
    # Send the input to the reservoir, char vector by char vector.
    sample[0].each do |vec|
        nR = reservoir.step(vec)
    end
    nRs << nR
    if index % (train_samples.size / 100) == 0 then
        puts "#{100*index / train_samples.size}%"
    end
end

puts "Learning..."


# learning
old_success = 0
epochs.times do |epoch|
    count = 0
    success = 0
    # puts "epoch=#{epoch}"
    train_samples.each.with_index do |sample,index|
        # # Restores the reservoir.
        # nR = reservoir.R = startR
        # # Send the input to the reservoir, char vector by char vector.
        # sample[0].each do |vec|
        #     nR = reservoir.step(vec)
        # end
        # # Use the reservoir as input for the NN.
        # nn.input(nR,sample[1]) # Input training data and teaching data.
        nn.input(nRs[index],sample[1]) # Input training data and teaching data.
        nn.run(1) # Compute feed forward propagation and backpropagation.

        count += 1
        res = nn.get_outputs.to_a.flatten
        if (sample[1].each_with_index.max[1] == res.each_with_index.max[1]) then
            success += 1
        end
        # puts "R=#{reservoir.R.to_a}"
        # puts "nR=#{nR.to_a}"
        # puts "nR=#{nR.to_a.flatten}"
        # puts "got output=#{nn.get_outputs.to_a.flatten}"
        # puts "expecting =#{sample[1]}"
        # if (count % 100 == 0) then
        #     puts "epoch=#{epoch} count=#{count} success=#{success} rate=#{success.to_f/count*100}%"
        # end
        # nn.compute_loss
        # loss << nn.get_loss
        # nn.initialize_loss
    end
    # Save learning state after each epoch if better success.
    if (success > old_success) then
        puts "Epoch=#{epoch}, better weights for rate=#{success.to_f/count*100}%"
        # nn.save_network("network_bnn.json") # save learned network
        old_success = success
    end
    # # Shall we regenerate the reservoir?
    # if epoch == 0 && success.to_f/count < 0.5  then
    #     # Yes.
    #     reservoir.generate_W
    #     startR = reservoir.R
    #     redo
    # end
end

puts "Understood!"

check_data = []
# confirmation of network
20.times do
    # Sample an input.
    index = rand(test_samples.size)
    sample = test_samples[index]
    # Send it to the reservoir.
    # Restores the reservoir.
    nR = reservoir.R = startR
    # Send the input to the reservoir, char vector by char vector.
    sample[0].each do |vec|
        nR = reservoir.step(vec)
    end
    # Use the reservoir as input for the NN.
    nn.input(nR)
    nn.propagate
    puts "exp   = #{sample[1]}"
    puts "guess = #{nn.get_outputs.to_a.flatten}"
end



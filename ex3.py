curr_dt = 2

import generate_data as d_gen
import numpy as np
if curr_dt == 0:
    print("importing line net")
    import nets
elif curr_dt == 1:
    print("importing parabola net")
    import nets_par as nets
elif curr_dt == 2:
    print("importing spiral net")
    import nets_spiral as nets
import matplotlib.pyplot as plt
import torch
import scipy.stats as sciline
FloatTensor = torch.FloatTensor

data_type = {0:"line", 1:"par", 2:"spiral"}
data_type_2_idx = {"line":0, "par":1, "spiral":2}
epochs = 10000#4000#300#2300
disc_steps = 20
gen_steps = 5
batch_size = 100#100#64
noise_vec_size = 20#20#16
d_lr = 0.001
g_lr = 0.001
test_size = 1000

def sample_z(sample_size):
    return np.random.normal(0, 1.25, (sample_size,noise_vec_size))

def train(data_type):
    global last_loss, patience, disc_steps, gen_steps
    real_labels = torch.ones((batch_size), dtype = torch.float)
    fake_labels = torch.zeros((batch_size), dtype = torch.float)
    for epoch in range(epochs):
        # ----- Train D ----- #
        for step in range(disc_steps):
            d_optim.zero_grad()
            noise_samples = FloatTensor(sample_z(batch_size))
            # get_data is already defined such that examples are already chosen from iid
            real_samples = FloatTensor(d_gen.get_data(batch_size, data_type))
            real_samp_decision = discriminator(real_samples)
            d_loss_real_samp = criterion(real_samp_decision.t(), real_labels)
            d_loss_real_samp.backward()

            gen_out = generator(noise_samples).detach()
            fake_samp_decision = discriminator(gen_out)
            d_loss_fake_samp = criterion(fake_samp_decision.t(), fake_labels)
            d_loss_fake_samp.backward()
            d_optim.step()
        # ----- Train G ----- #
        for step in range(gen_steps):
            g_optim.zero_grad()
            noise_samples = FloatTensor(sample_z(batch_size))
            gen_out = generator(noise_samples)
            fake_samp_decision = discriminator(gen_out)
            g_loss = criterion(fake_samp_decision.t(), real_labels)
            g_loss.backward()
            g_optim.step()
        if curr_dt == 2 and epoch == 1000:
            disc_steps -= 5
            gen_steps += 5
        d_err_avg = (d_loss_fake_samp.item() + d_loss_real_samp.item())/2
        print("Epoch %s, D loss real:%f, fake:%f, avg_err:%f , G %f err" % (epoch, d_loss_real_samp.item(),
                                                                       d_loss_fake_samp, d_err_avg, g_loss.item()))

        if epoch >1000 and epoch % 1000 == 0:
            test_spiral(epoch,d_err_avg,g_loss)
    return d_err_avg, g_loss

def init_params():
    discriminator = nets.Discriminator(2)
    generator = nets.Generator(noise_vec_size)
    disc_optim = torch.optim.Adam(discriminator.parameters(), lr = d_lr)
    gen_optim = torch.optim.Adam(generator.parameters(), lr=g_lr)
    return discriminator, generator, disc_optim, gen_optim, torch.nn.BCELoss()

def test_line(epoch = None, d_avg_loss = None, g_loss = None):
    epsilon = 10**-5
    gen_samples = get_test_points()

    corr = sum([gen_samples[i][0].item() == gen_samples[i][1].item() for i in range(test_size)])
    ep_corr = sum([abs(gen_samples[i][0].item() - gen_samples[i][1].item()) < epsilon for i in range(test_size)])
    print(sum([abs(gen_samples[i][0].item() - gen_samples[i][1].item()) < (10**-6) for i in range(test_size)]))
    x_axis, y_axis = plot_test(data_type[0], gen_samples, epoch, d_avg_loss, g_loss)
    _,_,r, _,_ = sciline.linregress(x_axis, y_axis)
    r_sqrt = r**2
    print("Tets acc: %f/%f, %f, ep_acc:%f/%f, %f, r_sqrt: %f" % (corr, test_size, corr*100/test_size,
                                                                 ep_corr, test_size, ep_corr*100/test_size, r_sqrt))

def get_test_points():
    noise_samples = FloatTensor(sample_z(test_size))
    return generator(noise_samples)

def test_par(epoch = None, d_avg_loss = None, g_loss = None):
    epsilon = 10**-5
    gen_samples = get_test_points()

    corr = sum([(gen_samples[i][0].item())**2 == gen_samples[i][1].item() for i in range(test_size)])
    print(sum([abs(gen_samples[i][0].item()**2 - gen_samples[i][1].item()) < epsilon for i in range(test_size)]))
    plot_test(data_type[1], gen_samples, epoch, d_avg_loss, g_loss)
    print("Tets acc: %f/%f, %f" % (corr, test_size, corr*100/test_size))

def test_spiral(epoch = None, d_avg_loss = None, g_loss = None):
    gen_samples = get_test_points()
    plot_test(data_type[2], gen_samples, epoch, d_avg_loss, g_loss)

def plot_test(data_type, gen_samples, epoch = None, d_avg_loss = None, g_loss = None):
    x_axis, y_axis = [], []
    [(x_axis.append(gen_samples[i][0].item()), y_axis.append(gen_samples[i][1].item())) for i in range(test_size)]
    plt.figure(figsize=(10,6))
    plt.plot(x_axis, y_axis, '.', label='class '+data_type)

    plt.legend()
    plt.grid(True)
    if epoch is not None and d_avg_loss is not None and g_loss is not None:
        plt.title('test set epoch %s, discriminator avg loss %f, generator loss %f'%(epoch,d_avg_loss,g_loss))
    else:
        plt.title('test set')
    plt.show()
    return x_axis, y_axis

if __name__ == '__main__':
    # ---- curr_dt is defined at the top of the script ----#
    discriminator, generator, d_optim, g_optim, criterion = init_params()
    d_avg_loss, g_loss = train(data_type[curr_dt])
    if curr_dt == 0:
        test_line(epochs, d_avg_loss, g_loss)
    elif curr_dt == 1:
        test_par(epochs, d_avg_loss, g_loss)
    elif curr_dt == 2:
         test_spiral(epochs, d_avg_loss, g_loss)


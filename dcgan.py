from utils.nets import *
from datas import *
import time
from tools import *


class DCGAN():
    def __init__(self, generator, discriminator, data):
        self.generator = generator
        self.discriminator = discriminator
        self.data = data

        # 噪声数据的参数
        self.z_dim = self.data.z_dim
        # 真实数据的参数
        self.size = self.data.size
        self.channel = self.data.channel

        # 输入的真实样本
        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        # 输入的噪声
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])

        # 生成器：G_sample就是生成的fake样本
        self.G_sample = self.generator(self.z)

        # 判别器
        # 判别真实的样本X
        self.D_real, _ = self.discriminator(self.X)
        # 判别生成的样本G_sample
        self.D_fake, _ = self.discriminator(self.G_sample, reuse=True)

        # loss函数
        # tf.reduce_mean()求均值,即期望E
        # tf.nn.sigmoid_cross_entropy_with_logits()求交叉熵：z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        # tf.ones_like()全为1的数组，大小和第一个参数一样，即labels=1，所以交叉熵中的z=1
        # tf.zeros_like()全为0的数组，大小和第一个参数一样，即labels=0，所以交叉熵中的z=0

        # 所以:
        # loss = E(-log(sigmoid(D_real))) + E(-log(1 - sigmoid(D_fake)))
        #          = - { E(log(sigmoid(D_real))) + E(log(1 - sigmoid(D_fake))) }
        self.D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(
            self.D_real))) + tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
        # loss = - E(log(sigmoid(D_fake)))
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))

        # 用自适应矩估计(adaptive moment estimation)算法对loss函数优化求解
        self.D_solver = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(self.D_loss, var_list=self.discriminator.vars)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(self.G_loss, var_list=self.generator.vars)

        # 保存变量
        self.saver = tf.train.Saver()

        # GPU配置
        gpu_options = tf.GPUOptions(allow_growth=True)

        # 获取Session
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        pass

    def train(self, sample_dir, ckpt_dir, training_epoches=1000000, batch_size=32):

        # 初始化变量
        self.sess.run(tf.global_variables_initializer())

        # 固定的噪声
        fixed_n = 5
        fixed_z = sample_z(fixed_n * fixed_n, self.z_dim)

        for epoch in range(training_epoches):
            # 更新D
            X_b = self.data(batch_size)
            self.sess.run(self.D_solver, feed_dict={self.X: X_b, self.z: sample_z(batch_size, self.z_dim)})

            # 更新G
            k = 1
            for _ in range(k):
                self.sess.run(self.G_solver, feed_dict={self.z: sample_z(batch_size, self.z_dim)})

            # save img, model. print loss
            if epoch % 100 == 0 or epoch < 100:
                # 当前的loss
                D_loss_curr = self.sess.run(self.D_loss, feed_dict={self.X: X_b, self.z: sample_z(batch_size, self.z_dim)})
                G_loss_curr = self.sess.run(self.G_loss, feed_dict={self.z: sample_z(batch_size, self.z_dim)})
                # 打印loss
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime((time.time()))),
                      'Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(epoch, D_loss_curr, G_loss_curr))

            # 保存图片
            if save_imgae(epoch=epoch):
                samples = self.sess.run(self.G_sample, feed_dict={self.z: fixed_z})

                # 保存生成的样本
                fig = self.data.data2fig(samples, size=fixed_n)
                plt.savefig('{}/{}.png'.format(sample_dir, str(epoch)), bbox_inches='tight')
                plt.close(fig)

            # 保存训练结果
            if epoch % (training_epoches//10) == 0 and epoch > 0:
                self.saver.save(self.sess, os.path.join(ckpt_dir, "dcgan-" + str(epoch) + ".ckpt"))

        pass

if __name__ == '__main__':

    # 生成的图片的目录
    sample_dir = 'Samples/celebA_dcgan'
    make_dir_if_noe_exist(sample_dir)
    ckpt_dir = 'Ckpts/celebA_dcgan'
    make_dir_if_noe_exist(ckpt_dir)

    # 生成器
    generator = G_conv()
    # 判别器
    discriminator = D_conv()
    # 数据
    data = celebA()
    # data = alisure()

    # 运行DCGAN
    dcgan = DCGAN(generator, discriminator, data)
    dcgan.train(sample_dir, ckpt_dir, training_epoches=500000)

    pass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt





def plot_n_recon(max_n,results):
    n_list = [i+1 for i in range(max_n)]
    plt.plot(n_list,results[0])
    plt.plot(n_list,results[1])
    plt.legend(['BLEU score with n < 6','BLEU score with n >= 6'])
    plt.savefig("./plotting_%d_recon.png"%max_n)
    return

def plot_n_recon_vs_1(max_n,results,results1,label1):
    n_list = [i+1 for i in range(max_n)]
    plt.plot(n_list,results[0])
    plt.plot(n_list,results[1])
    plt.plot(n_list,results1)
    plt.legend(['BLEU score with n < 6','BLEU score with n >= 6',label1])
    plt.savefig("./plotting_{}_recon_vs_{}.png".format(max_n,label1))
    return


def plot_n_recon_vs_2(max_n,results,results1,label1,results2,label2):
    n_list = [i+1 for i in range(max_n)]
    plt.plot(n_list,results[0])
    plt.plot(n_list,results[1])
    plt.plot(n_list,results1)
    plt.plot(n_list,results2)
    plt.legend(['BLEU score with n < 6','BLEU score with n >= 6',label1,label2])
    plt.savefig("./plotting_{}_recon_vs_{}_{}.png".format(max_n,label1,label2))
    return 

def plot_n_recon_vs_3(max_n,results,results1,label1,results2,label2,results3,label3):
    n_list = [i+1 for i in range(max_n)]
    plt.plot(n_list,results[0])
    plt.plot(n_list,results[1])
    plt.plot(n_list,results1)
    plt.plot(n_list,results2)
    plt.plot(n_list,result3)
    plt.legend(['BLEU score with n < 6','BLEU score with n >= 6',label1,label2,label3])
    plt.savefig("./plotting_{}_recon_vs_{}_{}_{}.png".format(max_n,label1,label2,label3))
    return

def plot_n_recon_vs_4(max_n,results,results1,label1,results2,label2,results3,label3,results4,label4):
    n_list = [i+1 for i in range(max_n)]
    plt.plot(n_list,results[0])
    plt.plot(n_list,results[1])
    plt.plot(n_list,results1)
    plt.plot(n_list,results2)
    plt.plot(n_list,result3)
    plt.plot(n_list,result4)
    plt.legend(['BLEU score with n < 6','BLEU score with n >= 6',label1,label2,label3,label4])
    plt.savefig("./plotting_{}_recon_vs_{}_{}_{}_{}.png".format(max_n,label1,label2,label3,label4))
    return





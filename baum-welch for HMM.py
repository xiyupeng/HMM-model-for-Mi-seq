import numpy as np
import math

#read the reference genome
def ref_genome(filename):
    readfile = open(filename, 'r')
    genome=""
    for line in readfile:
        line = line.strip('\n')
        if '>' in line:
            pass;
        else:
            genome=genome+line
    return genome

# read the reads mapping pattern X
def reads_mapping(filename,L):
    readfile = open(filename, 'r')
    X=L*[0]
    for line in readfile:
        line = line.strip('\n')
        line = line.split(' ')
        X[int(line[0])-1]=int(line[1])
    return X

# find the vector Z
def find_methy_candidate(genome,l_max):
    L=len(genome)
    Z=[]
    for i in range(0,L-l_max):
        n=genome[i:i+l_max+1].count("cg")
        z_v= n * [0]
        Z.append(z_v)
    nl=genome[L-l_max:L].count("cg")
    Z.append(nl*[0])
    return Z

# define the state space of Z with different length
def state_space(m_t):
    elelist = ["0", "1"]
    oldlist = elelist
    Omega = []
    if m_t==0:
        return [0]
    else:
        if m_t==1:
            return [[0],[1]]
        k = 1
        while k < m_t:
            Omega = []
            if m_t==1:
                return oldlist
            else:
                for item in oldlist:
                    for i in range(2):
                        Omega.append(item + elelist[i])
                oldlist = Omega
            k = k + 1
    for i in range(len(Omega)):
        newlist=[]
        for item in list(Omega[i]):
            newlist.append(int(item))
        Omega[i]=newlist
    return Omega

# find the norm of zt
def norm(zt):
    if zt==0:
        ztabs=0
    else:
        ztabs = np.linalg.norm(zt, ord=1)
    return ztabs

# possion
def p_poisson(lambda_n, x_n):
    p = (math.pow((math.e), lambda_n * (-1)) * math.pow(lambda_n, x_n)) / math.factorial(x_n)
    return p

#transiiton probability from Z to X, know lam, sig
def calPoisP(t,zt,lam,sig,X):
    ztabs = norm(zt)
    lam_t = lam * ztabs + sig
    x_t = X[t-1]
    q = p_poisson(lam_t, x_t)
    return (q)

#calculate the probabilty log(p(zt=V,x_1,x_2....x_L_lmax+1))
def cal_ztV(log_alpha_t,log_beta_t,V):
    if V==0:
        order=0
    else:
        m_t=len(V)
        order=state_space(m_t).index(V)
    return log_alpha_t[order]+log_beta_t[order]

#calculate the probabilty log(p(zt-1=V1,zt=V2, x_1,x_2....x_L_lmax+1))
def cal_Zt_V12(log_alpha_t_1,log_beta_t,t,V1, V2,lam,sig,p,X):
    if V1==0:
        order1=0
    else:
        m1=len(V1)
        order1 = state_space(m1).index(V1)
    if V2==0:
        order2=0
    else:
        m2=len(V2)
        order2=state_space(m2).index(V2)
    return log_alpha_t_1[order1]+math.log(caltransP(V2,V1,p)+0.000001)+calPoisP(t,V2,lam,sig,X)+log_beta_t[order2]

#calculate the conditional probabilty log(p(zt=V | x_1,x_2....x_L_lmax+1))
def conditional_ztV(log_alpha,log_beta,t,m_t):
    zt_space=state_space(m_t)
    mangi_p=len(zt_space)*[0]
    conditional_p=len(zt_space)*[0]
    scale = -(min(log_beta[t - 1])+min(log_alpha[t - 1]))  # positive number
    for i in range(len(zt_space)):
        mangi_p[i]=math.exp(cal_ztV(log_alpha[t-1],log_beta[t-1],zt_space[i])+scale)
    sum=np.sum(mangi_p)
    for j in range(len(zt_space)):
        conditional_p[j]=mangi_p[j]/sum
    return conditional_p

# caluclate the conditional probabilty log(p(zt-1=V1,zt=V2 | x_1,x_2....x_L_lmax+1))
def conditional_Zt_V12(log_alpha,log_beta,t,m_t,m_t_1,lam,sig,p,X):
    zt_space=state_space(m_t)
    zt_1_space=state_space(m_t_1)
    mangi_p=[([0] * len(zt_space)) for i in range(len(zt_1_space))] # len(zt_1_space)*len(zt_space) matrix
    conditional_p=[([0] * len(zt_space)) for i in range(len(zt_1_space))]
    scale=-(min(log_beta[t-1])+min(log_alpha[t-2]))#positive number
    for i in range(len(zt_1_space)):
        for j in range(len(zt_space)):
            mangi_p[i][j]=math.exp(cal_Zt_V12(log_alpha[t-2],log_beta[t-1],t,zt_1_space[i], zt_space[j],lam,sig,p,X)+scale)
    sum=np.sum(mangi_p)
    for k in range(len(zt_1_space)):
        for w in range(len(zt_space)):
            conditional_p[k][w]=mangi_p[k][w]/sum
    return conditional_p

#calculate p^(t+1) at each iteration step
def cal_methy_p(Z,X,lam,sig,p,log_alpha,log_beta):
    num_den=0         #denominator
    num_num= 0         #numeritor
    for t in range(2, len(Z)+1):
        if len(Z[t-1]) == len(Z[t - 2]) + 1:
            num_den = num_den + 1
            zt_1_space = state_space(len(Z[t-2]))  
            zt_space = state_space(len(Z[t-1]))     
            for i in range(len(zt_1_space)):
                for j in range(len(zt_space)):
                    if zt_1_space[i]==0:
                        zt_1_space[i]=[]
                    zt_1_space[i].append(1)
                    if zt_space[j]==zt_1_space[i]:
                        num_num = conditional_Zt_V12(log_alpha, log_beta,t,len(Z[t-1]),len(Z[t-2]), lam, sig, p, X)[i][j] + num_num
                        # len(zt_1_space)*len(zt_space) matrix[i][j]
                    zt_1_space = state_space(len(Z[t - 2]))
    return num_num/num_den

#calculate the expectation of norm(Z_t)
def cal_e_norm_zt(Z,log_alpha,log_beta):
    e_norm=[0] *len(Z)
    for t in range(1,len(Z)+1):
        zt_space = state_space(len(Z[t-1]))
        for i in range(len(zt_space)):
            e_norm[t-1]=e_norm[t-1]+norm(zt_space[i])*conditional_ztV(log_alpha,log_beta,t,len(Z[t-1]))[i]   #   m_t=len(Z[t])
    return e_norm

#calculate lambda^(t+1), sig^(t+1)
def cal_lam_sig(Z,X,lam,sig,log_alpha,log_beta):
    e_norm_zt=cal_e_norm_zt(Z,log_alpha,log_beta)
    new_lam=(np.sum(X)-len(X)*sig)/np.sum(e_norm_zt)
    new_sig=(np.sum(X)-lam*np.sum(e_norm_zt))/len(X)
    return[new_lam,new_sig]

# transition proabbility from z_t_1 to z_t
def caltransP(z_t,z_t_1,p):
    if (z_t==0 or z_t_1==0):
        return 1
    if(len(z_t)==len(z_t_1)):
        if (z_t==z_t_1):
            transP=1
        else:
            transP=0
    else:
        if(len(z_t)>len(z_t_1)):
            if (z_t_1==z_t[0:len(z_t)-1]):
                if (z_t[len(z_t)-1] == 1):
                    transP = p
                else:
                    transP = 1 - p
            else:
                transP=0
        else:
            if (z_t==z_t_1[1:len(z_t_1)]):
                transP=1
            else:
                transP=0
    return transP

#calculate alpha from beginning
def cal_alpha_list(Z,X,lam,sig,p):
    log_alpha_all=[]
    for t in range(1,len(Z)+1):
        log_alpha_all.extend(calalpha(log_alpha_all,t,p,Z,X,lam,sig))
    return  log_alpha_all

def calalpha(log_alpha_all,t,p,Z,X,lam,sig):
    log_alpha=[]
    if(t==1):
        Omega_1=state_space(len(Z[0]))
        alpha_0=[]
        for i in range(0,len(Omega_1)):
            n1=norm(Omega_1[i])
            ppois = calPoisP(t,Omega_1[i],lam,sig,X)
            if Omega_1[i]==0:
                p_z=1
            else:
                p_z=math.pow(p,n1)*(math.pow(1-p,len(Omega_1[i])-n1))
            v_i=ppois* p_z
            alpha_0.append(math.log(v_i))
        log_alpha.append(alpha_0)
    else:
        log_alpha_t=[]
        Omegapre = state_space(len(Z[t-2]))
        Omega = state_space(len(Z[t-1]))
        for vt in range (0, len(Omega)):
            PoissonP = calPoisP(t,Omega[vt],lam,sig,X)
            sum=0
            scale=-min(log_alpha_all[len(log_alpha_all)-1])  #positive number
            for v in range(0, len(Omegapre)):
                trans=caltransP(Omega[vt], Omegapre[v],p)
                log_alpha_t_1_v=log_alpha_all[len(log_alpha_all)-1][v]
                re=math.log(trans+0.000001)+log_alpha_t_1_v+math.log(PoissonP)
                re_s=re+scale
                sum=sum+math.exp(re_s)
            log_sum=math.log(sum)-scale
            log_alpha_t.append(log_sum)
        log_alpha.append(log_alpha_t)
    return  log_alpha

#calculate beta from end
def cal_beta_list(Z,X,lam,sig,p):
    log_beta_all=[]
    for t in range(1,len(Z)+1):
        log_beta_all.extend(calbeta(log_beta_all,len(Z)-t+1,p,Z,X,lam,sig))
    log_beta_all.reverse()
    return  log_beta_all

def calbeta(log_beta_all,t,p,Z,X,lam,sig):
    log_beta=[]
    if(t==len(Z)):
        Omega_l=state_space(len(Z[t-1]))
        log_beta_last=[]
        for i in range(0,len(Omega_l)):
            log_beta_last.append(0)
        log_beta.append(log_beta_last)
    else:
        log_beta_t=[]
        Omegapost = state_space(len(Z[t]))  #t+1
        Omega = state_space(len(Z[t-1]))    #t
        for vt in range (0, len(Omega)):
            sum=0
            scale = -min(log_beta_all[len(log_beta_all) - 1])
            for v in range(0, len(Omegapost)):
                PoissonP = calPoisP(t+1,Omegapost[v],lam,sig,X)
                trans = caltransP(Omegapost[v], Omega[vt],p)
                log_beta_t_1_v = log_beta_all[len(log_beta_all)-1][v]
                re = math.log(trans + 0.000001) + log_beta_t_1_v + math.log(PoissonP)
                re_s = re + scale
                sum = sum + math.exp(re_s)
            log_sum = math.log(sum) - scale
            log_beta_t.append(log_sum)
        log_beta.append(log_beta_t)
    return log_beta

#main
def implement(X,Z,genome):
    lam=1
    sig=1
    p=0.5
    stop=1
    while(stop>0.0001):
        log_alpha=cal_alpha_list(Z,X,lam,sig,p)## list
        log_beta = cal_beta_list(Z,X,lam,sig,p)## list
        new_p=cal_methy_p(Z, X, lam, sig, p, log_alpha, log_beta)
        #print(new_p)
        new_lam,new_sig=cal_lam_sig(Z,X,lam,sig,log_alpha,log_beta)
        #print(new_lam,new_sig)
        stop=abs((new_p-p)/p) ##
        #stop=abs((new_p-p)/p)
        p,lam,sig=new_p,new_lam,new_sig
        print(p,lam,sig)
    alpha=cal_alpha_list(Z,X,lam,sig,p)  #
    beta=cal_beta_list(Z,X,lam,sig,p)  #
    final_Z=[]
    for t in range(1,len(Z)+1):
        conditional_pro_vectr=conditional_ztV(alpha, beta, t, len(Z[t-1]))
        index=conditional_pro_vectr.index(max(conditional_pro_vectr))
        final_Z.append(state_space(len(Z[t-1]))[index])
    return [final_Z,p,lam,sig]

def countcg(genome,i,j):
    count=0
    for t in range(0,36):
        sub=genome[i+t:i+t+2]
        if(sub=="cg"):
            if count==j:
                pos=t+i+1
                return pos
            count=count+1
    return 0

l_max=36
test_genome=ref_genome("sim.genome.fa")
L=len(test_genome)
X=reads_mapping("summary.sim.36",L-l_max+1)
Z=find_methy_candidate(test_genome,l_max)
final_Z,p,lam,sig=implement(X,Z,test_genome)
print(p,lam,sig)
methy_position=[]

for i in range(len(final_Z)):
    if final_Z[i]==0:
        pass
    else:
        for j in range(len(final_Z[i])):
            if final_Z[i][j] == 1:
                methy_position.append(countcg(test_genome, i, j))
uniq_position=list(set(methy_position))
uniq_position.sort()
print("estimated methylated position")
print(uniq_position)
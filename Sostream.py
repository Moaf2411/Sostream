import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


class cluster:
    def __init__(self,id,centroid,num,radius,counter):
        self.id = id
        self.centroid = centroid
        self.num = num
        self.radius = radius
        self.counter = counter
        self.age = 0
        self.die = False




############################### distance ################################
def dist(a,b):
    return np.linalg.norm(a - b)







############################## mindist #################################
def mindist(clusters,v):
    d = np.Infinity
    winner = clusters[0]
    for c in clusters:
        dist1 = dist(v,c.centroid)
        if dist1 < d:
            d = dist1
            winner = c
    return winner






############################## find neighbors ##########################
def findneighbors(win,minpts,micros):
    dists = []
    ids = []
    if len(micros) > minpts:
        for m in micros:
            if m.id != win.id:
                dists.append(dist(win.centroid,m.centroid))
                ids.append(m)
        for j in range(len(dists)-1):
            for k in range(len(dists)-1):
                if dists[k] > dists[k+1]:
                    temp = dists[k]
                    dists[k] = dists[k+1]
                    dists[k+1] = temp
                    temp = ids[k]
                    ids[k] = ids[k+1]
                    ids[k+1] = temp
        win.radius = dists[minpts-1]
        return ids[0:minpts]

    else:
        return []







############################## update cluster ###########################
def updatecluster(win,v,alpha,winN):
    # update centroid of win 
    win.centroid = (win.centroid * win.num) + v
    win.centroid /= (win.num + 1)
    win.num += 1
    win.counter += 1
    rad = win.radius
    for n in winN:
        widthN = rad ** 2
        beta = np.exp(-1*(dist(win.centroid,n.centroid)/(2*widthN)))
        # adjust centroid
        n.centroid += alpha*beta*(win.centroid-n.centroid)








############################# new cluster ###############################
def newcluster(v,cid,micros):
    n = cluster(cid,v,1,0,1)
    micros.append(n)







############################ find overlap ##############################
def findoverlap(win,winN):
    overlap = []
    for n in winN:
        if dist(win.centroid,n.centroid) - (win.radius+n.radius) < 0:
            overlap.append(n)
    return overlap
    
    
    
    
    
    
    
    
 ######################### merge clusters #############################   
def mergeclusters(win,overlap,mergethreshold,micros):
    m = 0
    for o in overlap:
        if dist(win.centroid,o.centroid) < mergethreshold:
            new = ( (win.num * win.centroid) + (o.num * o.centroid) ) / (win.num+o.num)
            win.num += o.num
            win.radius = max( dist(new,win.centroid)+win.radius , dist(new,o.centroid)+o.radius )
            win.centroid = new
            m += 1
            o.die = True

    micros1 = []
    for s in micros:
        if s.die == False:
            micros1.append(s)

    return (m,micros1.copy())
            
    
    
  
    
  
############################# fade ########################################
def fadeall(clusters,fadethreshold,lam,time):
    fade = 0
    clusters1 = []
    for z in range(len(clusters)-1):
        f = 2 ** (lam*clusters[z].age)
        clusters[z].counter = clusters[z].counter * f
        if time % 10 == 0:
            if clusters[z].counter < fadethreshold:
                clusters[z].die=True
                fade += 1
            
    for c in clusters:
        if c.die == False:
            clusters1.append(c)
    return (fade,clusters1.copy())


    




############################### SOStream #################################
def sostream(ds,alpha,lam,minpts,mergethreshold,fadethreshold):
    cid = 0
    merged = 0
    faded = 0
    time = 0
    micros = []
    nums = []
    for i in ds:
      
        for q in micros:
            q.age += 1 
            
        time += 1
        (fade1,micros) = fadeall(micros,fadethreshold,lam,time)
        faded += fade1
        if len(micros) > minpts:
            win = mindist(micros,i)
            winN = findneighbors(win, minpts,micros)
            if dist(i,win.centroid) <= win.radius:
                updatecluster(win,i,alpha,winN)
            else:
                cid += 1
                newcluster(i,cid,micros)
            
            overlap = findoverlap(win,winN)
            if len(overlap) > 0:
                (merge,micros) = mergeclusters(win,overlap,mergethreshold,micros)
                merged += merge
            
        else:
            cid += 1 
            newcluster(i, cid,micros)
        nums.append(len(micros))
    
   
    return (micros,nums,merged,faded)
                








############################ test #######################################

def getcluster3d(a,b,c,res1):
    mini = dist(np.array([a,b,c]),res1[0])
    ind = 0
    for i in range(len(res1)):
        if dist(np.array([a,b,c]),res1[i]) < mini:
            mini = dist(np.array([a,b,c]),res1[i])
            ind = i
    return ind


def getcluster2d(a,b,res1):
    mini = dist(np.array([a,b]),res1[0])
    ind = 0
    for i in range(len(res1)):
        if dist(np.array([a,b]),res1[i]) < mini:
            mini = dist(np.array([a,b]),res1[i])
            ind = i
    return ind







def run(ds,alpha,lam,minpts,mergethreshold,fadethreshold,doplot=True):
    '''

    Parameters
    ----------
    ds : 
        path of the file containing the data stream. 
    alpha : 
        alpha.
    lam : 
        lambda parameter used to calculate fade.
    minpts : 
        MinPts. number of neighbors of the winner.
    mergethreshold : 
        threshold for merging.
    fadethreshold : 
        threshold for fading
    doplot :
        True => plot the results 
        False => do not plot the results
        
    Returns
    -------
    runs sostream on ds.
    
    micros : 
        list of final micro clusters.
    time:
        running time of algorithm for data stream ds.
    number of merged: 
        number of times that two clusters were merged.
    number of faded:
        number of clusters that faded.
    number of clusters:
        number of final clusters.
    plots:
        plot the number of clusters in each step
        plot of data points(for data streams with 2 or 3 features.) colored based on the cluster of points.
        saves the plots in the same folder.

    '''  
    

    data = pd.read_csv(ds)
    data = pd.read_csv(ds,names=np.arange(data.shape[1]))
    data = np.array(data,dtype='float64')
  
    
    
    import time
#   --------------------------------------
#   running sostream
    pretime = time.time_ns()
    (res,nums,merged,faded) = sostream(data,alpha,lam,minpts,mergethreshold,fadethreshold)
    aftertime = time.time_ns()
    res1 = []
    w = []
    radius1 = []
    for i in res:
        res1.append(i.centroid)
        w.append(i.num)
        radius1.append(i.radius)
        
        
#   saving the results as a csv file containing the centroids                   
    res = pd.DataFrame(res1,columns=np.arange(data.shape[1]))
    res['# points'] = w
    res['radius'] = radius1
    res.to_csv('results.csv',index=False)
    
#   --------------------------------------
    print('------------------------------------------------------')
    print('output:')
    print('running time:',end=' ')
    print((aftertime - pretime)/1000000000,end=' ')
    print('seconds')
#   ---------------------------------------
    print('number of clusters:',end=' ')
    print(len(res))
#   ---------------------------------------
    print('number of faded clusters:',end=' ')
    print(faded)
#   ---------------------------------------
    print('number of merged clusters:',end=' ')
    print(merged)





#   -------------------------------------
#   plotting cluster count in time steps
    fig1 = plt.figure(figsize=(10,5))
    ax = fig1.add_axes([0,0,1,1])
    
    ax.plot(np.arange(len(nums)),nums)
    fig1.savefig('cluster_count.jpg',bbox_inches='tight' )
#   ----------------------------------------

#   plotting data points
    
    colors = []
    for i in range(len(res)):
        color = []
        color.append(np.random.rand(1)[0])
        color.append(np.random.rand(1)[0])
        color.append(np.random.rand(1)[0])
        colors.append(color)
    
    
    
    if data.shape[1] == 3 and data.shape[0] <= 1000 and doplot:
        fig = plt.figure(dpi=200,figsize=(20,20))
        ax = fig.add_subplot(111,projection='3d')
        for i in data:
            color = colors[getcluster3d(i[0], i[1],i[2], res1)]
            ax.scatter(i[0],i[1],i[2],zdir='z',color=(color[0],color[1],color[2]),s=50)
        ax.scatter(res[0],res[1],res[2],zdir='z',c='k',s=300,alpha=0.5)
        fig.savefig('3d')
    
    if data.shape[1] == 2 and data.shape[0] <= 1000 and doplot:
        fig = plt.figure(dpi=200,figsize=(20,20))
        ax1 = fig.add_axes([0,0,1,1])
        for i in data:
            color = colors[getcluster2d(i[0], i[1], res1)]
            ax1.scatter(i[0],i[1],color=(color[0],color[1],color[2]),s=50)
        ax1.scatter(res[0],res[1],c='k',s=300,alpha=0.5)
        fig.savefig('2d.jpg',bbox_inches='tight')
        




########################################## run the algorithm ######################################################

add = input('enter the address of data stream file:')
add = add.replace('/','//')
alpha = float(input('enter the alpha parameter:'))
lam = float(input('enter the lambda parameter:'))
minpts = int(input('enter the minPts parameter:'))
mergethreshold = float(input('enter the merge threshold:'))
fadethreshold = float(input('enter the fade threshold:'))
doplot = int(input('enter 0 for False and 1 for True (for plotting the data points):'))           
run(add,alpha,lam,minpts,mergethreshold,fadethreshold,doplot)    




#run(ds='Dataset_2 .csv',alpha=0.1,lam=0.01,minpts=2,mergethreshold=2,fadethreshold=1.75,doplot=True)           
                
     
        

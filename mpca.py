import torch
def run(X, target_shape, eta=1e-3, iter=1024):
    shape = X.shape
    rank_num = len(X.shape)-1
    X_mean = X.mean(0)
    X_tilde = X-X_mean

    _X = []
    for i in range(1,rank_num+1):
        permute = [j for j in range(rank_num+1)]
        permute.remove(i)
        permute.insert(1,i)
        _X.append(X_tilde.permute(permute).reshape(X.shape[0],X.shape[i],-1))

    phi = [torch.bmm(_X[i], _X[i].permute(0,2,1)).sum(0) for i in range(rank_num)]

    eigens = []
    for i in range(rank_num):
        eigenvalues = torch.eig(phi[i],True).eigenvalues[:,0]
        eigenvectors = torch.eig(phi[i],True).eigenvectors
        _eigens = sorted([[eigenvalues[i], eigenvectors[i]] for i in range(len(eigenvalues))], key=lambda x:-x[0])[:target_shape[i]]
        vecs = torch.zeros(0)
        for val, vec in _eigens:vecs = torch.cat((vecs, vec.reshape(-1,1)),1)
        eigens.append(vecs)

    Y = X_tilde
    for i in range(1,rank_num+1):
        permute = [j for j in range(rank_num+1)]
        permute.remove(i)
        permute.append(i)
        Y = torch.matmul(Y.permute(permute), eigens[i-1])
        _permute = [permute.index(j) for j in range(rank_num+1)]
        Y = Y.permute(_permute)

    psi = sum([torch.norm(Y[i]) for i in range(Y.shape[0])])

    for k in range(iter):
        pre_Y = Y
        phi = []
        for n in range(rank_num):
            kron = torch.ones(1)
            for i in range(n+1,rank_num):kron = torch.kron(kron, eigens[i])
            for i in range(n):kron = torch.kron(kron, eigens[i])
            _phi = torch.zeros((shape[n+1],shape[n+1]))
            permute = [i for i in range(rank_num)]
            permute.remove(n)
            permute.insert(0,n)
            for x in X_tilde:
                x = x.reshape(x.shape[n],-1)
                x = x.mm(kron).mm(kron.T).mm(x.T)
                _phi += x
            phi.append(_phi)
        eigens = []
        for i in range(rank_num):
            eigenvalues = torch.eig(phi[i],True).eigenvalues[:,0]
            eigenvectors = torch.eig(phi[i],True).eigenvectors
            _eigens = sorted([[eigenvalues[i], eigenvectors[i]] for i in range(len(eigenvalues))], key=lambda x:-x[0])[:target_shape[i]]
            vecs = torch.zeros(0)
            for val, vec in _eigens:vecs = torch.cat((vecs, vec.reshape(-1,1)),1)
            eigens.append(vecs)
        Y = X_tilde
        for i in range(1,rank_num+1):
            permute = [j for j in range(rank_num+1)]
            permute.remove(i)
            permute.append(i)
            Y = torch.matmul(Y.permute(permute), eigens[i-1])
            _permute = [permute.index(j) for j in range(rank_num+1)]
            Y = Y.permute(_permute)
        pre_psi = sum([torch.norm(pre_Y[i]) for i in range(pre_Y.shape[0])])
        psi = sum([torch.norm(Y[i]) for i in range(Y.shape[0])])
        if pre_psi > psi:
            Y = pre_Y
            break
        elif psi - pre_psi < eta:break
    return Y
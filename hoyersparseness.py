import numpy as np

def Vecsparse(z):
    dim = len(z)
    argdim = np.sqrt(dim)
    l1 = np.sum(z)
    l2 = np.linalg.norm(z)
    sparse = (argdim-(l1/l2))/(argdim-1)
    return sparse

def MatrixSparse(A):
    dimA = A.shape[0]
    if type(A) == np.ndarray:
        pass
    else:
        A = A.toarray()
    allsparse = []
    for i in range(dimA):
        temp = Vecsparse(A[i])
        allsparse.append(temp)
    sparseM = np.mean(allsparse)
    return sparseM

def Zerochanger(stemp,dimx,ind,l1new,Z):
    stemp1 = np.copy(stemp)
    ind1 = np.where(stemp > 0)[0]
    cn = np.zeros(dimx)
    Z = np.append(Z, ind)
    stemp1[ind] = 0.
    c = (np.sum(stemp1) - l1new) / (dimx - len(Z))
    cn[ind1] = c
    snew = stemp1 - cn
    return snew, Z

def checkrange(valcheck,valrule,tol):
    valmin = valrule - tol*valrule
    valmax = valrule + tol*valrule
    if (valcheck >= valmin) and (valcheck <= valmax):
        return True
    else:
        return False

def operatorsparse(x,l1,l2,l1new,sparse,Z):
    stemp = np.array([])
    mtemp = np.array([])
    dimx = len(x)
    for i in range(dimx):
        temp = x[i] + (l1new - l1) / dimx
        stemp = np.append(stemp, temp)
        if i in Z:
            mtemp = np.append(mtemp, 0)
        else:
            mtemp = np.append(mtemp, l1new / (dimx - len(Z)))

    diffsm = stemp - mtemp
    p = np.sum(np.abs(diffsm))
    q = np.sum((mtemp * diffsm) * 2)
    r = np.sum(mtemp ** 2) - l2 ** 2
    akar = np.roots([p, q, r])
    rootsum = len(akar)
    roomsparse = np.array([])
    Ztemp = []
    Stemp = np.zeros((rootsum, dimx))
    for i in range(rootsum):
        si = mtemp + akar[i] * diffsm
        ind = np.where(si < 0)[0]
        if len(ind) > 0:
            sn, Zn = Zerochanger(si, dimx, ind, l1new, Z)
        else:
            sn = si
            Zn = Z
        Stemp[i] = sn
        Ztemp.append(Zn)
        sparsi = Vecsparse(sn)
        roomsparse = np.append(roomsparse, sparsi)
    ind = np.where((roomsparse - sparse) == np.min(roomsparse - sparse))[0][0]
    Z = np.append(Z,Ztemp[ind])
    sres = Stemp[ind]
    spres = roomsparse[ind]
    return sres, spres, Z


def VecOperatorsp(x, newsparse, tol):
    l1 = np.sum(x)
    l2 = np.linalg.norm(x)
    l1new = l2 * (np.sqrt(len(x)) - newsparse * (np.sqrt(len(x)) - 1))

    ind = [0]
    limit = 0
    while len(ind) > 0:
        Z = np.array([])
        newvec, sp_, Z = operatorsparse(x, l1, l2, l1new, newsparse, Z)
        l1 = np.sum(newvec)
        l2_ = np.linalg.norm(newvec)
        x = newvec
        statevar = [checkrange(l1, l1new, tol), checkrange(l2_, l2, tol), checkrange(sp_, newsparse, tol)]
        ind = np.where(np.array(statevar) != True)[0]
        limit = limit + 1

        if len(ind) <= 1 or limit == len(x):
            return newvec, sp_
            break

def MatrixOperatorSp(A,newsp,tol):
    dimA = A.shape[0]
    if type(A) == np.ndarray:
        pass
    else:
        A =  A.toarray()
    Asparse = MatrixSparse(A)
    if Asparse == newsp:
        print("This Matrix has same sparseness, indeed", Asparse)

    allsparse = np.array([])
    newA = np.copy(A)
    for i in range(dimA):
        vec,sp = VecOperatorsp(A[i],newsp,tol)
        allsparse = np.append(allsparse,sp)
        newA[i] = vec
    sparseM = np.mean(allsparse)
    return newA, sparseM
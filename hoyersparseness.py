import numpy as np

def Vecsparse(z):
    dim = len(z)
    if dim <= 1:
        return 0.0
    argdim = np.sqrt(dim)
    # Menggunakan np.abs untuk memastikan L1 norm yang valid
    l1 = np.sum(np.abs(z))
    l2 = np.linalg.norm(z)
    if l2 == 0:
        return 1.0  # Vektor nol dianggap sangat sparse
    sparse = (argdim - (l1 / l2)) / (argdim - 1)
    return sparse

def MatrixSparse(A):
    if not isinstance(A, np.ndarray):
        A = A.toarray()
    dimA = A.shape[0]
    allsparse = []
    for i in range(dimA):
        temp = Vecsparse(A[i])
        allsparse.append(temp)
    return np.mean(allsparse)

def Zerochanger(stemp, dimx, ind, l1new, Z):
    stemp1 = np.copy(stemp)
    ind1 = np.where(stemp > 0)[0]
    cn = np.zeros(dimx)
    
    # Gabungkan indeks unik ke Z
    Z = np.unique(np.append(Z, ind))
    stemp1[ind] = 0.0
    
    denom = dimx - len(Z)
    if denom <= 0:
        return stemp1, Z
        
    c = (np.sum(stemp1) - l1new) / denom
    cn[ind1] = c
    snew = stemp1 - cn
    return snew, Z

def checkrange(valcheck, valrule, tol):
    # Menggunakan batasan absolut toleransi agar lebih stabil
    return np.abs(valcheck - valrule) <= tol

def operatorsparse(x, l1, l2, l1new, sparse, Z):
    dimx = len(x)
    stemp = x + (l1new - l1) / dimx
    
    mtemp = np.zeros(dimx)
    denom = dimx - len(Z)
    if denom <= 0:
        denom = 1
    
    # Isi mtemp untuk indeks yang tidak ada di Z
    mask = np.ones(dimx, dtype=bool)
    if len(Z) > 0:
        mask[Z.astype(int)] = False
    mtemp[mask] = l1new / denom
    
    diffsm = stemp - mtemp
    p = np.sum(np.abs(diffsm))
    q = np.sum((mtemp * diffsm) * 2)
    r = np.sum(mtemp ** 2) - l2 ** 2
    
    akar = np.roots([p, q, r])
    # Ambil hanya akar riil untuk menghindari ComplexWarning
    akar = np.real(akar[np.isreal(akar)])
    
    if len(akar) == 0:
        # Fallback jika tidak ada akar riil
        akar = np.array([0.0])
        
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
        roomsparse = np.append(roomsparse, Vecsparse(sn))
    
    # Cari yang paling mendekati target 'sparse' dengan nilai absolut
    idx = np.argmin(np.abs(roomsparse - sparse))
    
    Z = np.unique(np.append(Z, Ztemp[idx]))
    sres = Stemp[idx]
    spres = roomsparse[idx]
    return sres, spres, Z

def VecOperatorsp(x, newsparse, tol):
    l1 = np.sum(x)
    l2 = np.linalg.norm(x)
    l1new = l2 * (np.sqrt(len(x)) - newsparse * (np.sqrt(len(x)) - 1))
    
    limit = 0
    max_iter = 100  # Pengaman anti-infinite loop
    
    while limit < max_iter:
        Z = np.array([])
        newvec, sp_, Z = operatorsparse(x, l1, l2, l1new, newsparse, Z)
        
        l1_curr = np.sum(newvec)
        l2_curr = np.linalg.norm(newvec)
        
        # Cek kondisi konvergensi
        c1 = checkrange(l1_curr, l1new, tol)
        c2 = checkrange(l2_curr, l2, tol)
        c3 = checkrange(sp_, newsparse, tol)
        
        x = newvec
        l1 = l1_curr
        l2 = l2_curr
        
        if c1 and c2 and c3:
            return newvec, sp_
            
        limit += 1
        
    return x, Vecsparse(x)

def MatrixOperatorSp(A, newsp, tol):
    if not isinstance(A, np.ndarray):
        A = A.toarray()
        
    dimA = A.shape[0]
    Asparse = MatrixSparse(A)
    
    if np.abs(Asparse - newsp) < tol:
        print("This Matrix has same sparseness, indeed", Asparse)
        return A, Asparse
        
    allsparse = np.array([])
    newA = np.copy(A).astype(float)
    
    for i in range(dimA):
        vec, sp = VecOperatorsp(A[i], newsp, tol)
        allsparse = np.append(allsparse, sp)
        newA[i] = vec
        
    sparseM = np.mean(allsparse)
    return newA, sparseM

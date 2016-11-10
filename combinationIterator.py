
class CombinationIterator():
    """Iterator for finding combinations without duplicates"""
    def __init__(self, lst, k):
        """Creates an iterator for the combinations C(n,k) from lst"""
        self.lst = lst
        self.n = len(lst)
        self.k = k
        self.indeces = list(range(k))
        #self.index_bmap = [1 if i < self.k else 0 for i in xrange(self.n)]
        self.done = False


    def __iter__(self):
        return self

    def __next__(self):
        if self.done:
            raise StopIteration
        else:
            comb = [self.lst[i] for i in self.indeces]

            overflow = True
            start = self.k-1
            while overflow and start >= 0:
                self.indeces[start] += 1
                if self.indeces[start] == self.n - (self.k - 1 - start):
                    overflow = True
                    start -= 1
                else:
                    overflow = False
            if overflow and start < 0:
                self.done = True
                return comb
            start += 1
            while start < self.k:
                self.indeces[start] = self.indeces[start-1] + 1
                start += 1

            return comb



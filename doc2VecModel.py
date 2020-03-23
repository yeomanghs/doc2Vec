#Load module
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity

#doc2Vec model
class similarDoc():
    def __init__(self, **kwargs):
        #predefine prameters
        self.Doc2VecModel = Doc2Vec(**kwargs)
        self.MaxEpoch = 10
        self.ModelName = datetime.now().strftime('%Y-%m-%d') + '_' + 'Doc2Vec.model'
        #indices are based on list of tagged document
        self.BadStrIndex = kwargs['bad_index']
        self.Threshold = kwargs['threshold']
        self.VectorSize = kwargs['vector_size']
        self.SimilarBadDocList = []
        
    def prepareText(self, ContentList):
        #tag document
#         self.TaggedDocList = [TaggedDocument(doc, tags = [str(no)]) 
#                               for no, doc in enumerate([i.split() for i in ContentList])]
        self.TaggedDocList = [TaggedDocument(doc[1], tags = [doc[0]]) 
                              for doc in zip(self.BadStrIndex, [i.split() for i in ContentList])]
    def trainModel(self):
        #build vocab
        self.Doc2VecModel.build_vocab(self.TaggedDocList)
        for epoch in range(self.MaxEpoch):
            print('Iteration - %s'%epoch)
            self.Doc2VecModel.train(self.TaggedDocList,
                                    total_examples = self.Doc2VecModel.corpus_count,
                                    epochs = epoch)
            #self.Doc2VecModel.alpha -= 0.002
        self.Doc2VecModel.save(self.ModelName)
        print('%s is saved'%self.ModelName)
        
    def loadModel(self, TrainedModel):
        self.Doc2VecModel = gensim.models.doc2vec.Doc2Vec.load(TrainedModel)
                    
    def findSimilarDocForBadDoc(self, NewDoc, n = 10):
        NewVec = self.Doc2VecModel.infer_vector(NewDoc.split())
        similarRepList = []
        similarScoreList = []
        #loop through every bad report
        for index in self.BadStrIndex:
            similarScore = cosine_similarity(self.Doc2VecModel.docvecs[index].reshape(1, self.VectorSize),
                                             NewVec.reshape(1, self.VectorSize))
            #if similarity score exceeds threshold
            if similarScore >= self.Threshold:
                similarRepList.append(str(index))
                similarScoreList.append(str(similarScore[0][0]))
        #if not similar to any bad report, (a, b, c)
        #a - 1 if has similar to any bad report
        #b - list of reports 
        if len(similarRepList) == 0:
            return (0, '', '')
        else:
            return (1, ','.join(similarRepList), ','.join(similarScoreList))

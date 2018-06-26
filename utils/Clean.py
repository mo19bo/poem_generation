import kenlm
import jieba

class Clean():
    def __init__(self,model_path):
        self.model = kenlm.LanguageModel(model_path)

    def _perplexity(self,sentence):
        words = len(sentence.split()) + 1
        return 10.0 ** (-self.model.score(sentence) / words)

    def _cut(self,stri):
        cut_str = jieba.cut(stri)
        list_str = [word for word in cut_str]
        stri = ' '.join(list_str)
        return stri

    def ppl(self,sentence):
        text = str(sentence)
        return self._perplexity(self._cut(text))


    def clean_dialog(self,text_list,thr=4):
        def _diff(list_a, list_b):
            ret_list = list(set(list_a) ^ set(list_b))
            return len(ret_list)

        s=[]
        for t in text_list:
            s.append(t.strip('\n'))

        s = [_ for _ in s if _ != '']
        index = 0
        res = []
        sim = [s[index]]

        while index < len(s) - 1:
            if s[index].replace(' ', '') == '':
                res.append(s[index])
                index += 1
                sim = [s[index]]

            elif _diff(s[index], s[index + 1]) < thr:
                sim.append(s[index + 1])
                index += 1

            elif len(sim) > 1:
                min = 100000000
                nice = []
                for text in sim:
                    try:
                        dnn_text = text.replace(' ', '')

                        ppl_res = self.ppl(dnn_text)
                        if ppl_res < min:
                            min = ppl_res
                            nice.append(text)
                    except Exception as e:
                        print('API调用错误：')
                        print(str(e))
                        continue

                res.append(nice[-1])
                index += 1
                sim = [s[index]]

            else:
                res.append(sim[0])
                index += 1
                sim = [s[index]]

        if len(sim) > 0:  # sim中还有残留
            res.append(sim[0])

        return res


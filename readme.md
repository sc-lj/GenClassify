使用文本生成方式进行分类:
```
<extra_id_0><extra_id_0>分类<extra_id_5>'具体类别1'<extra_id_0>情感<extra_id_5>正/负<extra_id_1><extra_id_1><extra_id_0>分类<extra_id_5>'具体类别2'<extra_id_0>情感<extra_id_5>正/负<extra_id_1><extra_id_1><extra_id_1>
```

使用文本生成方式进行分类:
```
<extra_id_0><extra_id_0>'具体类别1'<extra_id_5>正/中/负<extra_id_1><extra_id_0>'具体类别2'<extra_id_5>正/中/负<extra_id_1><extra_id_1>
```

schema格式:
```
["价格优劣势", "价格公正性", "优惠活动"]
["正", "中", "负"]
```

数据格式，是csv文件；
```
退票退款这个费劲,"[{'sentiment': -1, 'label': ['服务及时性', '退费及时性'], 'frag': '退票退款这个费劲'}]"
```
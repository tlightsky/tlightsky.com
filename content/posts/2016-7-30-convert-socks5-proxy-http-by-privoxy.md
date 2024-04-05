---
layout: post
title:  "convert sokcs5 proxy http by privoxy"
date:   2016-07-30
categories: sockes http proxy privoxy
---

## privoxy

安装privoxy
{% highlight shell %}
brew install privoxy
{% endhighlight %}

编辑配置文件
`vi ~/config`

{% highlight shell %}
listen-address 0.0.0.0:1080
forward-socks5 / localhost:1082 .
{% endhighlight %}


运行
{% highlight shell %}
/usr/local/Cellar/privoxy/3.0.24/sbin/privoxy
{% endhighlight %}

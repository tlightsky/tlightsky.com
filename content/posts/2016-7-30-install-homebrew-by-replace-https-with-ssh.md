---
layout: post
title:  "install homebrew by replace https with ssh"
date:   2016-07-30
categories: homebrew git https ssh
---

## homebrew 下载https出错

根据[github上的issue](https://github.com/Homebrew/legacy-homebrew/issues/34363)
描述，只需修改`~/.gitconfig`，添加以下内容


{% highlight shell %}
[url "git@github.com:"]
    insteadOf = "https://github.com/"
{% endhighlight %}

如果你的git@github.com也不ok的话，请参考：[github socks5 proxy](http://tlightsky.github.io/github/socke5/proxy/2016/07/25/github-socks5-proxy.html)

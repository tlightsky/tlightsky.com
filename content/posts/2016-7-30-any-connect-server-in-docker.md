---
layout: post
title:  "any connect server in docker"
date:   2016-07-30 
categories: vpn any connect
---

## 基本上适用docker即可

先克隆这个项目：
[github docker](https://github.com/wppurking/ocserv-docker)

然后写docker-compose.yml

{% highlight yml %}
any:
  image: wppurking/ocserv
  ports:
   - "443:443/tcp"
  privileged: true
  volumes:
   - ./ocserv:/etc/ocserv
  restart: always
{% endhighlight %}

初始化好的两个账户: wyatt:616 holly:525

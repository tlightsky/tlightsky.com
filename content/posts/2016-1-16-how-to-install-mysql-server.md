---
layout: post
title:  "如何安装mysql服务"
date:   2016-01-16
categories: wordpress mysql linux
---


执行命令`apt-get install mysql-server -y`，
执行过程中需要输入root的初始密码

如果需要从非本地访问数据库，需要做一些修改：

* 修改`/etc/mysql/my.cnf`，注释掉`bind-address            = 127.0.0.1`，然后重启mysql

* 添加一个root的远程账户，代码如下：

{% highlight bash %}
mysql -u root -p

use mysql;
update user set host='%' where user='root';
GRANT ALL PRIVILEGES ON *.* TO 'root'@'%' WITH GRANT OPTION;
flush privileges;
{% endhighlight %}


* 备份mysql

{% highlight bash %}
mysql dump -uroot -p dbname > name.sql
{% endhighlight %}


* 导入mysql

{% highlight bash %}
mysql -uroot -p dbname < name.sql
{% endhighlight %}

* 修密码

{% highlight bash %}
update user set password=password('123') where user='root' and host='localhost';
{% endhighlight %}

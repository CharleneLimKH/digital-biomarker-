# generate ssh key


Generate the private and public key with the following command in the terminal

    ssh-keygen -t ed25519 -C "charlene.lim.kh@gmail.com"

It will ask you if the path is correct. Just press enter and keep it like this. for password also.

    Generating public/private ed25519 key pair.
    Enter file in which to save the key (C:\Users\charl/.ssh/id_ed25519):

When it's done you will see a message like this

    Your identification has been saved in C:\Users\charl/.ssh/id_ed25519_github
    Your public key has been saved in C:\Users\charl/.ssh/id_ed25519_github.pub
    The key fingerprint is:
    SHA256: ..... charlene.lim.kh@gmail.com
    The key's randomart image is:
    +--[ED25519 256]--+
    |                 |
    |                 |
    |            .    |
    |         . o ..  |
    |        S = .+o o|
    |         * @.*EBB|
    |        . @.@oB*X|
    |         + *+=.+.|
    |        . o+=..  |
    +----[SHA256]-----+

To activate the key type the following command

    start-ssh-agent

    # you will get this message back 
    Removing old ssh-agent sockets
    Starting ssh-agent:  done

Then you can add the public key to github. The key is locaded in the folder C:\Users\charl\.ssh\<id>.pub.

# To force push

1. If git ask to pull first
    git pull

2. merge after pull
    git push --set-upstream digibm main   # git push --set-upstream <project> <branch>

3. push force
    git push --force-with-lease